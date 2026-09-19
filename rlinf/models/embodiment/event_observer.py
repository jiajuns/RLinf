"""Causal Event Observer and Event Value Critic sidecar modules.

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
    state_logits: torch.Tensor
    boundary_logits: torch.Tensor
    progress: torch.Tensor
    uncertainty: torch.Tensor


class EventObserver(nn.Module):
    """Predict event state from causal visual geometry and proprioception.

    The GRU is intentionally unidirectional: output at timestep ``t`` is not
    allowed to consume future frames.  ``mount_tokens`` distinguishes wrist
    and global-camera feature bundles without exposing embodiment identifiers.
    """

    def __init__(
        self,
        feature_dim: int,
        num_event_posteriors: int,
        num_smdp_states: int,
        *,
        hidden_dim: int = 256,
        num_mount_tokens: int = 3,
    ) -> None:
        super().__init__()
        if min(feature_dim, num_event_posteriors, num_smdp_states, hidden_dim, num_mount_tokens) < 1:
            raise ValueError("EventObserver dimensions must be positive")
        self.feature_dim = feature_dim
        self.input_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.mount_embedding = nn.Embedding(num_mount_tokens, hidden_dim)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.posterior_head = nn.Linear(hidden_dim, num_event_posteriors)
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


def event_observer_supervision_loss(
    prediction: EventObserverPrediction,
    *,
    posterior_target: torch.Tensor,
    state_target: torch.Tensor,
    boundary_target: torch.Tensor,
    progress_target: torch.Tensor,
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
    return {
        "posterior": (posterior * mask).sum() / denominator,
        "state": (state * mask).sum() / denominator,
        "boundary": (boundary * mask).sum() / denominator,
        "progress": (progress * mask).sum() / denominator,
    }
