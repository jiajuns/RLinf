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

import torch
from torch import nn
from torch.nn import functional as F

from rlinf.algorithms.event_credit import event_smdp_credit
from rlinf.models.embodiment.event_observer import EventObserver, EventValueCritic


@dataclass(frozen=True)
class EventSidecarOutput:
    """Causal sidecar predictions aligned to a ``[batch,time]`` rollout."""

    representation: torch.Tensor
    values: torch.Tensor
    event_ids: torch.Tensor
    boundary_probability: torch.Tensor
    progress: torch.Tensor
    uncertainty: torch.Tensor


class OnlineEventValueSidecar(nn.Module):
    """Frozen observer plus an online-updated SMDP Event Value Critic."""

    def __init__(self, observer: EventObserver, event_value: EventValueCritic) -> None:
        super().__init__()
        self.observer = observer
        self.event_value = event_value
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

    def freeze_observer(self) -> None:
        self.observer.eval()
        for parameter in self.observer.parameters():
            parameter.requires_grad_(False)

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
