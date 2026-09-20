"""Online Event Value sidecar for frozen Role-Graph Event Observers.

This component is deliberately outside the π0.5 actor/critic.  It loads an
offline-trained observer, freezes it, predicts causal event boundaries and
Event Values during rollout, then updates only ``V_E`` from current-policy
rollouts.  This avoids the policy-dependent Event Value becoming stale while
also ensuring PPO never backpropagates into SAM or simulator oracle labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from rlinf.algorithms.event_credit import event_smdp_credit
from rlinf.models.embodiment.event_observer import EventObserver, EventValueCritic, RGBRoleFeatureStudent


@dataclass(frozen=True)
class EventSidecarOutput:
    """Causal sidecar predictions aligned to a ``[batch,time]`` rollout."""

    representation: torch.Tensor
    values: torch.Tensor
    event_ids: torch.Tensor
    boundary_probability: torch.Tensor
    progress: torch.Tensor
    uncertainty: torch.Tensor


@dataclass(frozen=True)
class EventSidecarRollout:
    """Sidecar tensors aligned to RLinf's chunked embodied rollout layout.

    ``event_ids`` has ``[num_chunks, batch, action_chunk]`` shape and
    ``event_values`` has the corresponding bootstrap shape
    ``[num_chunks + 1, batch, action_chunk]``.  The observer sees only one
    RGB observation at each policy-chunk boundary (the information actually
    available in RoboTwin's current vectorised API), then its causal state is
    repeated across actions executed by that chunk.  ``action_representation``
    is the same expansion with one final bootstrap representation and is used
    solely for the online Event Value update.
    """

    event_ids: torch.Tensor
    event_values: torch.Tensor
    action_representation: torch.Tensor
    chunk_output: EventSidecarOutput


class OnlineEventValueSidecar(nn.Module):
    """Frozen observer plus an online-updated SMDP Event Value Critic."""

    def __init__(
        self,
        observer: EventObserver,
        event_value: EventValueCritic,
        rgb_student: RGBRoleFeatureStudent | None = None,
        *,
        proprio_time_delta: float = 0.1,
    ) -> None:
        super().__init__()
        self.observer = observer
        self.event_value = event_value
        self.rgb_student = rgb_student
        if proprio_time_delta <= 0:
            raise ValueError("proprio_time_delta must be positive")
        self.proprio_time_delta = proprio_time_delta
        self.freeze_observer()

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: torch.device | str = "cpu") -> "OnlineEventValueSidecar":
        checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
        required = {"observer", "event_value", "feature_dim", "posterior_dim", "state_dim", "geometric_dim", "state_change_dim"}
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"Event Observer checkpoint lacks {sorted(missing)}")
        observer = EventObserver(
            int(checkpoint["feature_dim"]), int(checkpoint["posterior_dim"]), int(checkpoint["state_dim"]),
            num_geometric_primitives=int(checkpoint["geometric_dim"]),
            num_state_change_primitives=int(checkpoint["state_change_dim"]),
        )
        observer.load_state_dict(checkpoint["observer"])
        value = EventValueCritic(256)
        value.load_state_dict(checkpoint["event_value"])
        return cls(observer, value).to(device)

    @classmethod
    def from_checkpoints(
        cls, event_checkpoint: str | Path, rgb_student_checkpoint: str | Path, *, device: torch.device | str = "cpu"
    ) -> "OnlineEventValueSidecar":
        """Load a deployable RGB→Event Observer→Event Value sidecar."""
        sidecar = cls.from_checkpoint(event_checkpoint, device=device)
        checkpoint = torch.load(Path(rgb_student_checkpoint), map_location=device, weights_only=True)
        if "rgb_student" not in checkpoint or int(checkpoint.get("feature_dim", -1)) != sidecar.observer.feature_dim:
            raise ValueError("RGB student checkpoint is incompatible with Event Observer feature_dim")
        student = RGBRoleFeatureStudent(sidecar.observer.feature_dim)
        student.load_state_dict(checkpoint["rgb_student"])
        sidecar.rgb_student = student.to(device)
        for parameter in sidecar.rgb_student.parameters():
            parameter.requires_grad_(False)
        sidecar.rgb_student.eval()
        return sidecar

    def freeze_observer(self) -> None:
        self.observer.eval()
        for parameter in self.observer.parameters():
            parameter.requires_grad_(False)
        if self.rgb_student is not None:
            self.rgb_student.eval()
            for parameter in self.rgb_student.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def infer_images(
        self,
        head_images: torch.Tensor,
        wrist_images: torch.Tensor,
        measured_state16: torch.Tensor | None = None,
        *,
        boundary_threshold: float = 0.5,
    ) -> EventSidecarOutput:
        """Run the online RGB student then the frozen Event Observer."""
        if self.rgb_student is None:
            raise RuntimeError("infer_images requires an RGB student checkpoint")
        return self.infer(
            self.rgb_student(
                head_images,
                wrist_images,
                measured_state16,
                proprio_time_delta=self.proprio_time_delta,
            ),
            boundary_threshold=boundary_threshold,
        )

    @torch.no_grad()
    def infer(
        self, features: torch.Tensor, mount_tokens: torch.Tensor | None = None, *, boundary_threshold: float = 0.5
    ) -> EventSidecarOutput:
        """Infer event IDs and ``V_E`` without exposing oracle labels online.

        ``features`` must be produced by a deployment-capable RGB/proprio
        frontend, never by an HDF/Zarr lookup.  A boundary starts a new SMDP
        event; the cumulative ID is local to each rollout sequence.
        """
        if not 0.0 < boundary_threshold < 1.0:
            raise ValueError("boundary_threshold must be strictly between zero and one")
        prediction = self.observer(features, mount_tokens)
        boundary_probability = prediction.boundary_logits.sigmoid()
        boundaries = boundary_probability >= boundary_threshold
        boundaries[:, 0] = True
        event_ids = boundaries.to(torch.long).cumsum(dim=1) - 1
        return EventSidecarOutput(
            representation=prediction.representation,
            values=self.event_value(prediction.representation),
            event_ids=event_ids,
            boundary_probability=boundary_probability,
            progress=prediction.progress,
            uncertainty=prediction.uncertainty,
        )

    def smdp_value_loss(
        self,
        representation: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        event_ids: torch.Tensor,
        *,
        gamma: float,
    ) -> torch.Tensor:
        """TD-style Event-SMDP loss on current-policy rollout data.

        Tensor layout here is time-major ``[T,B]`` for rewards/IDs and
        ``[T+1,B]`` for dones/values.  Targets are detached, use real event
        duration discounting, and use a zero influence tensor because this
        critic update estimates value rather than action credit.
        """
        if representation.ndim != 3:
            raise ValueError("representation must be [batch,time_plus_one,dim]")
        values = self.event_value(representation).transpose(0, 1)
        if values.shape != dones.shape or rewards.shape != event_ids.shape or values.shape[0] != rewards.shape[0] + 1:
            raise ValueError("Event Value rollout tensors have inconsistent time-major shapes")
        _, targets = event_smdp_credit(
            rewards, dones, event_ids, values.detach(), torch.zeros_like(rewards), gamma=gamma
        )
        return F.smooth_l1_loss(values[:-1], targets.detach())


def _right_wrist_stream(wrist_images: torch.Tensor) -> torch.Tensor:
    """Select the right wrist stream from a time-major RLinf observation.

    RoboTwin exposes wrist streams as ``[T,B,W,H,W,3]`` in left-to-right
    order.  The task-matched RGB student is trained with its right-camera HDF
    stream, so online inference must use the matching stream rather than
    silently concatenate both cameras.  Single-wrist embodiments use
    ``[T,B,H,W,3]`` and pass through unchanged.
    """
    if wrist_images.ndim == 6:
        if wrist_images.shape[-1] != 3 or wrist_images.shape[2] < 1:
            raise ValueError("wrist_images must be [time,batch,camera,height,width,3]")
        return wrist_images[:, :, -1]
    if wrist_images.ndim == 5 and wrist_images.shape[-1] == 3:
        return wrist_images
    raise ValueError("wrist_images must be [time,batch,height,width,3] or include a camera dimension")


def infer_event_sidecar_rollout(
    sidecar: OnlineEventValueSidecar,
    curr_obs: Mapping[str, torch.Tensor],
    next_obs: Mapping[str, torch.Tensor],
    *,
    num_action_chunks: int,
    boundary_threshold: float = 0.5,
) -> EventSidecarRollout:
    """Infer learned Event-SMDP fields from real online RoboTwin RGB.

    This function intentionally consumes the rollout observations, not the
    offline SAM/Zarr cache.  ``curr_obs`` and ``next_obs`` are time-major
    tensors recorded by :class:`EmbodiedTrajectoryBuilder`; their final next
    observation supplies the exact bootstrap image needed by ``V_E``.  It is
    kept framework-independent so the actor integration is directly unit
    testable without creating FSDP workers.
    """
    if num_action_chunks < 1:
        raise ValueError("num_action_chunks must be positive")
    required = {"main_images", "wrist_images"}
    missing = required.difference(curr_obs) | required.difference(next_obs)
    if missing:
        raise ValueError(f"online Event Value requires rollout observations {sorted(missing)}")
    current_head = curr_obs["main_images"]
    current_wrist = _right_wrist_stream(curr_obs["wrist_images"])
    next_head = next_obs["main_images"]
    next_wrist = _right_wrist_stream(next_obs["wrist_images"])
    if current_head.ndim != 5 or current_head.shape[-1] != 3:
        raise ValueError("main_images must have shape [time,batch,height,width,3]")
    if next_head.shape != current_head.shape or next_wrist.shape != current_wrist.shape:
        raise ValueError("current and next RGB observations must have equal time-major shapes")
    if current_head.shape[:2] != current_wrist.shape[:2]:
        raise ValueError("main_images and wrist_images must share time/batch dimensions")

    # The GRU is batch-first.  Append only the final successor: every earlier
    # successor is the next current frame and would otherwise be duplicated.
    head = torch.cat((current_head, next_head[-1:]), dim=0).transpose(0, 1).contiguous()
    wrist = torch.cat((current_wrist, next_wrist[-1:]), dim=0).transpose(0, 1).contiguous()
    measured = None
    if "measured_state16" in curr_obs and "measured_state16" in next_obs:
        current_measured = curr_obs["measured_state16"]
        next_measured = next_obs["measured_state16"]
        if current_measured.shape[:2] != current_head.shape[:2] or next_measured.shape != current_measured.shape:
            raise ValueError("measured_state16 is not aligned to current/next observations")
        measured = torch.cat((current_measured, next_measured[-1:]), dim=0).transpose(0, 1).contiguous()
    output = sidecar.infer_images(head, wrist, measured, boundary_threshold=boundary_threshold)
    chunks, batch = current_head.shape[:2]
    if output.values.shape != (batch, chunks + 1):
        raise RuntimeError("Event sidecar output is not aligned to chunk observations")

    # Flattened action-level advantage code expects the values in a padded
    # [C+1,B,K] layout.  The last bootstrap is placed at index C,B,0; the
    # remaining padded values are never consumed by preprocess_embodied_... .
    event_ids = output.event_ids[:, :-1].transpose(0, 1).unsqueeze(-1).expand(
        chunks, batch, num_action_chunks
    ).contiguous()
    event_values = output.values.new_zeros((chunks + 1, batch, num_action_chunks))
    event_values[:-1] = output.values[:, :-1].transpose(0, 1).unsqueeze(-1).expand(
        chunks, batch, num_action_chunks
    )
    event_values[-1, :, 0] = output.values[:, -1]
    action_representation = torch.cat(
        (
            output.representation[:, :-1].repeat_interleave(num_action_chunks, dim=1),
            output.representation[:, -1:],
        ),
        dim=1,
    )
    return EventSidecarRollout(
        event_ids=event_ids,
        event_values=event_values,
        action_representation=action_representation,
        chunk_output=output,
    )


@torch.no_grad()
def infer_branch_future_event_values(
    sidecar: OnlineEventValueSidecar,
    curr_obs: Mapping[str, torch.Tensor],
    branch_main_images: torch.Tensor,
    branch_wrist_images: torch.Tensor,
    branch_measured_state16: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate ``V_E(z_{t+H})`` for sparse matched-state branch endpoints.

    Branch endpoints cannot be evaluated as isolated frames: the causal GRU
    state must first see the exact online RGB prefix that existed at branch
    time.  For every policy chunk this helper repeats that prefix for each
    candidate, appends its returned endpoint image, and reads the final Event
    Value.  The small number of selected branch states makes this explicit
    implementation preferable to silently resetting temporal state.
    """
    required = {"main_images", "wrist_images"}
    missing = required.difference(curr_obs)
    if missing:
        raise ValueError(f"branch Event Value requires current observations {sorted(missing)}")
    head = curr_obs["main_images"]
    wrist = _right_wrist_stream(curr_obs["wrist_images"])
    if head.ndim != 5 or head.shape[-1] != 3:
        raise ValueError("current main_images must be [time,batch,height,width,3]")
    # branch wrist is [T,B,M,(camera),H,W,3].  Align it to the right-camera
    # convention used by the RGB student.
    if branch_wrist_images.ndim == 7:
        branch_wrist = branch_wrist_images[:, :, :, -1]
    elif branch_wrist_images.ndim == 6:
        branch_wrist = branch_wrist_images
    else:
        raise ValueError("branch wrist RGB has invalid shape")
    if branch_main_images.ndim != 6 or branch_main_images.shape[-1] != 3:
        raise ValueError("branch main RGB must be [time,batch,candidates,height,width,3]")
    chunks, batch, candidates = branch_main_images.shape[:3]
    if head.shape[:2] != (chunks, batch) or branch_wrist.shape[:3] != (chunks, batch, candidates):
        raise ValueError("branch RGB tensors are not aligned to current rollout observations")

    values = []
    for chunk_idx in range(chunks):
        # [B,M,T,H,W,3] -> [B*M,T,H,W,3]
        prefix_head = head[: chunk_idx + 1].permute(1, 0, 2, 3, 4)
        prefix_wrist = wrist[: chunk_idx + 1].permute(1, 0, 2, 3, 4)
        repeated_head = prefix_head[:, None].expand(-1, candidates, -1, -1, -1, -1)
        repeated_wrist = prefix_wrist[:, None].expand(-1, candidates, -1, -1, -1, -1)
        endpoint_head = branch_main_images[chunk_idx].unsqueeze(2)
        endpoint_wrist = branch_wrist[chunk_idx].unsqueeze(2)
        sequence_head = torch.cat((repeated_head, endpoint_head), dim=2).flatten(0, 1)
        sequence_wrist = torch.cat((repeated_wrist, endpoint_wrist), dim=2).flatten(0, 1)
        sequence_measured = None
        if branch_measured_state16 is not None:
            if "measured_state16" not in curr_obs:
                raise ValueError("branch measured state needs current measured_state16")
            current_measured = curr_obs["measured_state16"]
            if branch_measured_state16.shape[:3] != (chunks, batch, candidates):
                raise ValueError("branch measured state is not aligned to branch images")
            prefix_measured = current_measured[: chunk_idx + 1].permute(1, 0, 2)
            repeated_measured = prefix_measured[:, None].expand(-1, candidates, -1, -1)
            endpoint_measured = branch_measured_state16[chunk_idx].unsqueeze(2)
            sequence_measured = torch.cat((repeated_measured, endpoint_measured), dim=2).flatten(0, 1)
        endpoint_values = sidecar.infer_images(sequence_head, sequence_wrist, sequence_measured).values[:, -1]
        values.append(endpoint_values.reshape(batch, candidates))
    return torch.stack(values, dim=0)
