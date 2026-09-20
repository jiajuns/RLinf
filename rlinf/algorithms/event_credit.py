"""Semi-Markov event credit for πRL PPO.

This module deliberately changes only the advantage estimator.  Flow-SDE
sampling, the π0.5 policy, PPO ratios and PPO losses remain RLinf's own.
"""
from __future__ import annotations

import torch

from rlinf.algorithms.registry import register_advantage
from rlinf.algorithms.utils import safe_normalize


def _check_inputs(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    event_ids: torch.Tensor,
    event_values: torch.Tensor,
    influence: torch.Tensor,
) -> None:
    if rewards.ndim != 2:
        raise ValueError("event credit expects flattened [T,B] rewards")
    steps, batch = rewards.shape
    if dones.shape != (steps + 1, batch):
        raise ValueError("dones must be [T+1,B]")
    if event_ids.shape != (steps, batch) or influence.shape != (steps, batch):
        raise ValueError("event ids and influence must be [T,B]")
    if event_values.shape != (steps + 1, batch):
        raise ValueError("event values must be [T+1,B]")


def event_smdp_credit(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    event_ids: torch.Tensor,
    event_values: torch.Tensor,
    intervention_influence: torch.Tensor,
    *,
    gamma: float,
    influence_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build SMDP targets and allocate each event advantage by intervention.

    ``event_ids[t]`` identifies the event active while action ``t`` is executed.
    Each contiguous id run is an event, hence reused ids must begin a new run.
    ``event_values[t]`` is :math:`V_E(z_t^E)` and influence is
    :math:`I(s_t,a_t)=Y(s_t,a_t)-mean_a Y(s_t,a)` from same-state branches.

    Event j obtains ``R_j + gamma**D_j * V_E(next) - V_E(start)``.  The
    interventional allocation preserves the *sign* of the measured influence:
    an action that lowers future Event Value receives negative PPO credit even
    inside an otherwise successful event.  Its magnitude is normalized by the
    event's total absolute influence.  This intentionally differs from a
    softmax: softmax can only assign positive weights and therefore quietly
    rewards harmful actions.

    An all-zero influence run is the temporal/uniform ablation.  In that case
    every action receives the shared Event-SMDP advantage instead of a silent
    zero-gradient event.
    """
    _check_inputs(rewards, dones, event_ids, event_values, intervention_influence)
    if not 0 < gamma <= 1 or influence_temperature <= 0:
        raise ValueError("invalid gamma or influence temperature")
    steps, batch = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = event_values[:-1].clone()
    for b in range(batch):
        start = 0
        while start < steps:
            if event_ids[start, b] < 0:
                start += 1
                continue
            event = event_ids[start, b]
            end = start + 1
            while end < steps and event_ids[end, b] == event and not dones[end, b]:
                end += 1
            duration = end - start
            discounts = rewards.new_tensor(gamma).pow(torch.arange(duration, device=rewards.device))
            event_reward = (discounts * rewards[start:end, b]).sum()
            terminal = bool(dones[end, b])
            bootstrap = event_values[end, b] if not terminal else event_reward.new_zeros(())
            event_return = event_reward + (gamma**duration) * bootstrap
            event_advantage = event_return - event_values[start, b]
            scaled = intervention_influence[start:end, b] / influence_temperature
            absolute = scaled.abs()
            denominator = absolute.sum()
            if bool(denominator <= torch.finfo(absolute.dtype).eps):
                # Uniform temporal Event-SMDP allocation used by the no-branch
                # ablation, and a stable fallback before an Influence Model has
                # accumulated useful branch supervision.
                advantages[start:end, b] = event_advantage
            else:
                allocation = duration * absolute / denominator
                advantages[start:end, b] = (
                    allocation * scaled.sign() * event_advantage.abs()
                )
            returns[start:end, b] = event_values[start:end, b] + advantages[start:end, b]
            start = end
    return advantages, returns


@register_advantage("event_smdp_interventional")
def compute_event_smdp_interventional_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    event_ids: torch.Tensor | None = None,
    event_values: torch.Tensor | None = None,
    intervention_influence: torch.Tensor | None = None,
    gamma: float = 1.0,
    normalize_advantages: bool = True,
    loss_mask: torch.Tensor | None = None,
    influence_temperature: float = 1.0,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if event_ids is None or event_values is None or intervention_influence is None:
        raise ValueError("event_smdp_interventional requires event_ids, event_values and intervention_influence")
    advantages, returns = event_smdp_credit(
        rewards, dones, event_ids, event_values, intervention_influence,
        gamma=gamma, influence_temperature=influence_temperature,
    )
    if normalize_advantages:
        advantages = safe_normalize(advantages, loss_mask=loss_mask)
    return advantages, returns


@register_advantage("event_smdp_temporal")
def compute_event_smdp_temporal_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    event_ids: torch.Tensor | None = None,
    event_values: torch.Tensor | None = None,
    gamma: float = 1.0,
    normalize_advantages: bool = True,
    loss_mask: torch.Tensor | None = None,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Oracle-event SMDP target with uniform within-event temporal credit.

    This is a named ablation, not the proposed interventional estimator: the
    influence is identically zero until same-state Flow-SDE branches are wired.
    """
    if event_ids is None or event_values is None:
        raise ValueError("event_smdp_temporal requires event_ids and event_values")
    advantages, returns = event_smdp_credit(
        rewards,
        dones,
        event_ids,
        event_values,
        torch.zeros_like(event_ids, dtype=event_values.dtype),
        gamma=gamma,
    )
    if normalize_advantages:
        advantages = safe_normalize(advantages, loss_mask=loss_mask)
    return advantages, returns
