"""Task-general causal Event Observer and Event Value Critic sidecars.

These modules consume only cached visual geometry/tracking features and
deployment-time proprioception.  SAM-derived masks are produced offline by a
separate teacher pipeline; neither SAM nor simulator oracle state is part of
the PPO actor update.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EventObserverPrediction:
    """Causal Event Observer outputs aligned to an input feature sequence."""

    representation: torch.Tensor
    posterior_logits: torch.Tensor
    geometric_relation_logits: torch.Tensor
    state_change_logits: torch.Tensor
    state_logits: torch.Tensor
    boundary_logits: torch.Tensor
    progress: torch.Tensor
    uncertainty: torch.Tensor


class EventObserver(nn.Module):
    """Predict shared relation primitives before a task-general event state.

    The GRU is intentionally unidirectional: output at timestep ``t`` is not
    allowed to consume future frames.  ``mount_tokens`` distinguishes wrist
    and global-camera feature bundles without exposing embodiment identifiers.
    Task/archetype IDs are deliberately absent: only cached role-pair features,
    role types, and deployment-time proprioception may condition this module.
    """

    def __init__(
        self,
        feature_dim: int,
        num_event_posteriors: int,
        num_smdp_states: int,
        *,
        hidden_dim: int = 256,
        num_mount_tokens: int = 3,
        num_geometric_primitives: int | None = None,
        num_state_change_primitives: int = 1,
    ) -> None:
        super().__init__()
        if min(feature_dim, num_event_posteriors, num_smdp_states, hidden_dim, num_mount_tokens) < 1:
            raise ValueError("EventObserver dimensions must be positive")
        if num_geometric_primitives is None:
            num_geometric_primitives = num_event_posteriors
        if num_geometric_primitives < 1 or num_state_change_primitives < 0:
            raise ValueError("relation primitive dimensions must be non-negative")
        self.feature_dim = feature_dim
        self.input_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.mount_embedding = nn.Embedding(num_mount_tokens, hidden_dim)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.posterior_head = nn.Linear(hidden_dim, num_event_posteriors)
        self.geometric_relation_head = nn.Linear(hidden_dim, num_geometric_primitives)
        self.state_change_head = nn.Linear(hidden_dim, num_state_change_primitives)
        self.state_head = nn.Linear(hidden_dim, num_smdp_states)
        self.boundary_head = nn.Linear(hidden_dim, 1)
        self.progress_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        mount_tokens: torch.Tensor | None = None,
    ) -> EventObserverPrediction:
        """Return causal event predictions for ``features`` of shape ``[B,T,D]``."""
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must have shape [batch,time,{self.feature_dim}], got {tuple(features.shape)}"
            )
        projected = self.input_projection(features)
        if mount_tokens is not None:
            if mount_tokens.shape != features.shape[:2]:
                raise ValueError("mount_tokens must have shape [batch,time]")
            projected = projected + self.mount_embedding(mount_tokens.to(torch.long))
        representation, _ = self.temporal(projected)
        return EventObserverPrediction(
            representation=representation,
            posterior_logits=self.posterior_head(representation),
            geometric_relation_logits=self.geometric_relation_head(representation),
            state_change_logits=self.state_change_head(representation),
            state_logits=self.state_head(representation),
            boundary_logits=self.boundary_head(representation).squeeze(-1),
            progress=torch.sigmoid(self.progress_head(representation).squeeze(-1)),
            uncertainty=F.softplus(self.uncertainty_head(representation).squeeze(-1)) + 1e-6,
        )


class EventValueCritic(nn.Module):
    """Estimate ``V_E(z_t^E)`` from a learned causal event representation."""

    def __init__(self, representation_dim: int, *, hidden_dim: int = 256) -> None:
        super().__init__()
        if representation_dim < 1 or hidden_dim < 1:
            raise ValueError("EventValueCritic dimensions must be positive")
        self.value = nn.Sequential(
            nn.LayerNorm(representation_dim),
            nn.Linear(representation_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        """Return values with shape ``representation.shape[:-1]``."""
        if representation.ndim < 2:
            raise ValueError("representation must include batch and feature dimensions")
        return self.value(representation).squeeze(-1)


class RGBRoleFeatureStudent(nn.Module):
    """Small online RGB student for offline SAM teacher feature vectors.

    SAM 3.1 remains an offline teacher only.  This model consumes the head and
    wrist RGB streams already supplied to π0.5, projects them to the compact
    Role-Graph feature interface, and lets a frozen Event Observer run on new
    PPO states without HDF/Zarr lookup.
    """

    proprio_feature_dim = 12

    def __init__(self, feature_dim: int, *, hidden_dim: int = 128, image_size: tuple[int, int] = (96, 128)) -> None:
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1 or min(image_size) < 16:
            raise ValueError("RGBRoleFeatureStudent dimensions must be positive")
        self.feature_dim = feature_dim
        self.image_size = image_size
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim + self.proprio_feature_dim),
            nn.Linear(hidden_dim + self.proprio_feature_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    @staticmethod
    def measured_proprio_features(
        measured_state16: torch.Tensor, *, time_delta: float = 0.1
    ) -> torch.Tensor:
        """Derive causal deployable gripper/EE motion features from readback.

        Layout is left TCP pose (7), left aperture, right TCP pose (7), right
        aperture.  The output exactly occupies the nonvisual tail of the
        offline teacher interface: normalized aperture, causal aperture delta,
        two TCP linear deltas and their speeds.  It deliberately does not read
        any action command, depth, oracle mask, or object state.
        """
        if measured_state16.ndim != 3 or measured_state16.shape[-1] != 16 or time_delta <= 0:
            raise ValueError("measured_state16 must be [batch,time,16]")
        opening = measured_state16[..., (7, 15)].clamp(0.0, 1.0)
        position = torch.stack((measured_state16[..., :3], measured_state16[..., 8:11]), dim=2)
        opening_delta = torch.zeros_like(opening)
        position_delta = torch.zeros_like(position)
        opening_delta[:, 1:] = (opening[:, 1:] - opening[:, :-1]) / time_delta
        position_delta[:, 1:] = (position[:, 1:] - position[:, :-1]) / time_delta
        speed = torch.linalg.vector_norm(position_delta, dim=-1)
        return torch.cat((opening, opening_delta, position_delta.flatten(2), speed), dim=-1)

    def _prepare_images(self, images: torch.Tensor, name: str) -> torch.Tensor:
        if images.ndim != 5 or images.shape[-1] != 3:
            raise ValueError(f"{name} must have shape [batch,time,height,width,3]")
        batch, steps = images.shape[:2]
        value = images.reshape(batch * steps, *images.shape[2:]).permute(0, 3, 1, 2).float()
        if value.max() > 1.0:
            value = value / 255.0
        return F.interpolate(value, size=self.image_size, mode="bilinear", align_corners=False)

    def forward(
        self,
        head_images: torch.Tensor,
        wrist_images: torch.Tensor,
        measured_state16: torch.Tensor | None = None,
        proprio_time_delta: float = 0.1,
    ) -> torch.Tensor:
        if head_images.shape[:2] != wrist_images.shape[:2]:
            raise ValueError("head and wrist image batch/time prefixes must match")
        batch, steps = head_images.shape[:2]
        if measured_state16 is None:
            proprio = torch.zeros(
                (batch, steps, self.proprio_feature_dim), device=head_images.device, dtype=torch.float32
            )
        else:
            if measured_state16.shape[:2] != (batch, steps):
                raise ValueError("measured_state16 must share image batch/time prefixes")
            proprio = self.measured_proprio_features(
                measured_state16.to(torch.float32), time_delta=proprio_time_delta
            )
        encoded = self.encoder(torch.cat((
            self._prepare_images(head_images, "head_images"),
            self._prepare_images(wrist_images, "wrist_images"),
        ), dim=1))
        encoded = encoded.reshape(batch, steps, -1)
        return self.head(torch.cat((encoded, proprio), dim=-1))


def event_observer_supervision_loss(
    prediction: EventObserverPrediction,
    *,
    posterior_target: torch.Tensor,
    state_target: torch.Tensor,
    boundary_target: torch.Tensor,
    progress_target: torch.Tensor,
    geometric_relation_target: torch.Tensor | None = None,
    state_change_target: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute masked posterior, state, boundary, and uncertainty-aware progress losses."""
    if posterior_target.shape != prediction.posterior_logits.shape:
        raise ValueError("posterior_target shape must match posterior_logits")
    if state_target.shape != prediction.state_logits.shape[:2]:
        raise ValueError("state_target must have shape [batch,time]")
    if boundary_target.shape != prediction.boundary_logits.shape:
        raise ValueError("boundary_target shape must have shape [batch,time]")
    if progress_target.shape != prediction.progress.shape:
        raise ValueError("progress_target shape must have shape [batch,time]")
    if valid_mask is None:
        valid_mask = torch.ones_like(prediction.progress, dtype=torch.bool)
    if valid_mask.shape != prediction.progress.shape or not bool(valid_mask.any()):
        raise ValueError("valid_mask must contain at least one [batch,time] observation")
    mask = valid_mask.to(prediction.progress.dtype)
    denominator = mask.sum()
    posterior = F.binary_cross_entropy_with_logits(
        prediction.posterior_logits, posterior_target.to(prediction.posterior_logits.dtype), reduction="none"
    ).mean(dim=-1)
    state = F.cross_entropy(
        prediction.state_logits.flatten(0, 1), state_target.flatten().to(torch.long), reduction="none"
    ).reshape_as(mask)
    boundary = F.binary_cross_entropy_with_logits(
        prediction.boundary_logits, boundary_target.to(prediction.boundary_logits.dtype), reduction="none"
    )
    squared_error = (prediction.progress - progress_target.to(prediction.progress.dtype)).square()
    progress = 0.5 * (squared_error / prediction.uncertainty + prediction.uncertainty.log())
    losses = {
        "posterior": (posterior * mask).sum() / denominator,
        "state": (state * mask).sum() / denominator,
        "boundary": (boundary * mask).sum() / denominator,
        "progress": (progress * mask).sum() / denominator,
    }
    for name, logits, target in (
        ("geometric_relations", prediction.geometric_relation_logits, geometric_relation_target),
        ("state_changes", prediction.state_change_logits, state_change_target),
    ):
        if target is None:
            continue
        if target.shape != logits.shape:
            raise ValueError(f"{name} target shape must match its relation head")
        per_frame = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), reduction="none")
        losses[name] = (per_frame.mean(dim=-1) * mask).sum() / denominator
    return losses
