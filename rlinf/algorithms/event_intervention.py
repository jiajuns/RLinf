"""Same-state short-rollout intervention utilities for Event-SMDP credit.

The policy still samples actions through RLinf's π0.5 Flow-SDE rollout path.
This module only evaluates those already sampled actions from an identical
simulator snapshot. An adapter must explicitly expose snapshot/restore; a
normal reset is not a controlled intervention.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import torch
from torch import nn


class SnapshotEnvironment(Protocol):
    """Minimum simulator contract required by a controlled action branch."""

    def get_state(self) -> Any:
        """Return a complete cloneable simulator state."""

    def load_state(self, state: Any) -> None:
        """Restore a state emitted by :meth:`get_state`."""


BranchRollout = Callable[[Any], tuple[torch.Tensor, torch.Tensor, int]]


class EventInfluenceModel(nn.Module):
    """Amortize sparse matched-state branch evidence over all event actions.

    The model is intentionally small and independent of the π0.5 actor.  Its
    inputs are an online-capable observer representation, the executed action,
    and optional deployment-time state features.  Real simulator branches
    supervise it with centered ``I(s,a,z^E)`` values; PPO then queries it at
    every action in the event without running additional branches.
    """

    def __init__(
        self,
        event_representation_dim: int,
        action_dim: int,
        *,
        state_dim: int = 0,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if min(event_representation_dim, action_dim, hidden_dim) < 1 or state_dim < 0:
            raise ValueError("Influence Model dimensions must be valid")
        self.event_representation_dim = event_representation_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.network = nn.Sequential(
            nn.LayerNorm(event_representation_dim + action_dim + state_dim),
            nn.Linear(event_representation_dim + action_dim + state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        event_representation: torch.Tensor,
        actions: torch.Tensor,
        state_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if event_representation.shape[:-1] != actions.shape[:-1]:
            raise ValueError("event representation and action prefixes must match")
        if event_representation.shape[-1] != self.event_representation_dim or actions.shape[-1] != self.action_dim:
            raise ValueError("Influence Model input dimensions do not match construction")
        values = [event_representation, actions]
        if self.state_dim:
            if state_features is None or state_features.shape != (*actions.shape[:-1], self.state_dim):
                raise ValueError("state_features must match the configured prefix and state_dim")
            values.append(state_features)
        elif state_features is not None:
            raise ValueError("state_features were passed but state_dim is zero")
        # The upstream π0.5/FSDP stack can set its default floating dtype to
        # float64 while rollout actions and the frozen RGB Observer remain
        # float32.  ``I_ξ`` is an auxiliary network, so its own parameter dtype
        # is the unambiguous interface contract; normalize its inputs here
        # instead of relying on a process-wide default dtype.
        inputs = torch.cat(values, dim=-1)
        parameter = next(self.network.parameters())
        return self.network(inputs.to(device=parameter.device, dtype=parameter.dtype)).squeeze(-1)

    @staticmethod
    def loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError("influence prediction and target shapes must match")
        error = (prediction - target).square()
        if mask is None:
            return error.mean()
        if mask.shape != prediction.shape or not bool(mask.any()):
            raise ValueError("influence supervision mask must match and select data")
        return error.masked_select(mask.bool()).mean()


def policy_relative_influence(
    scorer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    event_representation: torch.Tensor,
    action: torch.Tensor,
    reference_actions: torch.Tensor,
) -> torch.Tensor:
    """Score an action relative to same-state policy action samples.

    State-centered Influence training identifies ranking only up to an
    arbitrary state-specific offset.  PPO must therefore consume
    ``f(z,a) - mean_m f(z,a_ref_m)`` rather than raw ``f(z,a)``.  Reference
    chunks are sampled from the current Flow-SDE policy but are *not*
    simulator branches, so they add inference cost but no environment
    interactions.  Prefix is arbitrary (normally ``[batch,chunks]``).
    """
    if event_representation.shape[:-1] != action.shape[:-1]:
        raise ValueError("event representation/action prefixes must match")
    if reference_actions.ndim != action.ndim + 1 or reference_actions.shape[:-2] != action.shape[:-1]:
        raise ValueError("reference actions must be [..., candidates, action_dim]")
    if reference_actions.shape[-1] != action.shape[-1] or reference_actions.shape[-2] < 1:
        raise ValueError("reference candidate dimension/action width is invalid")
    score = scorer(event_representation, action)
    candidates = reference_actions.shape[-2]
    reference_representation = event_representation.unsqueeze(-2).expand(*event_representation.shape[:-1], candidates, event_representation.shape[-1])
    reference_score = scorer(reference_representation, reference_actions)
    if score.shape != action.shape[:-1] or reference_score.shape != reference_actions.shape[:-1]:
        raise ValueError("Influence scorer returned incompatible score shapes")
    return score - reference_score.mean(dim=-1)


def _restore_snapshot(environment: SnapshotEnvironment, state: Any) -> None:
    """Use RLinf's real ManiSkill snapshot API, retaining legacy test adapters."""
    restore = getattr(environment, "load_state", None)
    if restore is None:
        # ``set_state`` is retained only for light-weight third-party adapters;
        # ManiSkillOffloadEnv exposes ``load_state`` because its snapshot is a
        # serialized complete state rather than a raw simulator tensor.
        restore = getattr(environment, "set_state", None)
    if restore is None:
        raise TypeError("controlled branches require load_state() or set_state()")
    restore(state)


def branch_event_returns(
    environment: SnapshotEnvironment,
    actions: Sequence[Any],
    rollout: BranchRollout,
    *,
    gamma: float,
) -> torch.Tensor:
    """Evaluate candidate actions from one state and restore the live state.

    ``rollout`` returns ``(discounted_rewards, future_event_value,
    executed_steps)`` after a short horizon. The return for each action is
    ``R + gamma**H * V_E(z_{t+H})``.
    """
    if not actions or not 0 < gamma <= 1:
        raise ValueError("branches and gamma must be nonempty and valid")
    snapshot = environment.get_state()
    values: list[torch.Tensor] = []
    try:
        for action in actions:
            _restore_snapshot(environment, snapshot)
            rewards, future_value, executed_steps = rollout(action)
            if executed_steps < 0:
                raise ValueError("branch rollout returned a negative horizon")
            values.append(rewards + (gamma**executed_steps) * future_value)
    finally:
        _restore_snapshot(environment, snapshot)
    return torch.stack(values)


def intervention_influence(branch_returns: torch.Tensor) -> torch.Tensor:
    """Center future Event Values across candidates from one saved state."""
    if branch_returns.ndim != 1 or branch_returns.numel() < 2:
        raise ValueError("intervention needs at least two returns from one state")
    return branch_returns - branch_returns.mean()
