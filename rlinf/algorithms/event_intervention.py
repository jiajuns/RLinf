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


class SnapshotEnvironment(Protocol):
    """Minimum simulator contract required by a controlled action branch."""

    def get_state(self) -> Any:
        """Return a complete cloneable simulator state."""

    def set_state(self, state: Any) -> None:
        """Restore a state emitted by :meth:`get_state`."""


BranchRollout = Callable[[Any], tuple[torch.Tensor, torch.Tensor, int]]


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
            environment.set_state(snapshot)
            rewards, future_value, executed_steps = rollout(action)
            if executed_steps < 0:
                raise ValueError("branch rollout returned a negative horizon")
            values.append(rewards + (gamma**executed_steps) * future_value)
    finally:
        environment.set_state(snapshot)
    return torch.stack(values)


def intervention_influence(branch_returns: torch.Tensor) -> torch.Tensor:
    """Center future Event Values across candidates from one saved state."""
    if branch_returns.ndim != 1 or branch_returns.numel() < 2:
        raise ValueError("intervention needs at least two returns from one state")
    return branch_returns - branch_returns.mean()
