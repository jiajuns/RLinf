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
    influence_beta: float = 0.0,
    influence_clip: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build credit-conserving SMDP targets and within-event redistribution.

    ``event_ids[t]`` identifies the event active while action ``t`` is executed.
    Each contiguous id run is an event, hence reused ids must begin a new run.
    ``event_values[t]`` is :math:`V_E(z_t^E)` and influence is
    :math:`I(s_t,a_t)=Y(s_t,a_t)-mean_a Y(s_t,a)` from same-state branches.

    Event j obtains ``R_j + gamma**D_j * V_E(next) - V_E(start)``.  The
    The Event-SMDP advantage determines an event's *total* PPO credit.  The
    intervention signal may only redistribute this credit within the event;
    it never decides the total sign.  This is deliberately more conservative
    than the historical signed-|I| allocator: a sparsely trained Influence
    Model must not be able to turn a successful SFT action into a negative PPO
    update merely by predicting the wrong sign.

    For an event of duration ``D`` the base allocation is ``A_E / D``.  The
    centered, clipped influence residual has zero sum, so
    ``sum_t A_t == A_E`` up to floating point error:

    ``A_t = A_E/D + beta * |A_E| * centered_zscore(I_t)``.

    With ``influence_beta=0`` (the default) this is the uniform Event-SMDP
    ablation.  Callers combine this Event credit with GAE through
    :func:`compute_event_smdp_residual_advantages`; this function itself does
    not silently replace a stable action critic.
    """
    _check_inputs(rewards, dones, event_ids, event_values, intervention_influence)
    if not 0 < gamma <= 1 or influence_temperature <= 0:
        raise ValueError("invalid gamma or influence temperature")
    if influence_beta < 0 or influence_clip <= 0:
        raise ValueError("influence_beta must be nonnegative and influence_clip positive")
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
            base = event_advantage / duration
            advantages[start:end, b] = base
            scaled = intervention_influence[start:end, b] / influence_temperature
            centered = scaled - scaled.mean()
            std = centered.std(unbiased=False)
            if influence_beta > 0 and bool(std > torch.finfo(centered.dtype).eps):
                redistribution = (centered / (std + torch.finfo(centered.dtype).eps)).clamp(
                    -influence_clip, influence_clip
                )
                # Clipping can change the mean; recentering restores exact
                # event-credit conservation.
                redistribution = redistribution - redistribution.mean()
                advantages[start:end, b] = base + influence_beta * event_advantage.abs() * redistribution
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
    influence_beta: float = 0.0,
    influence_clip: float = 3.0,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if event_ids is None or event_values is None or intervention_influence is None:
        raise ValueError("event_smdp_interventional requires event_ids, event_values and intervention_influence")
    advantages, returns = event_smdp_credit(
        rewards, dones, event_ids, event_values, intervention_influence,
        gamma=gamma,
        influence_temperature=influence_temperature,
        influence_beta=influence_beta,
        influence_clip=influence_clip,
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


@register_advantage("event_smdp_residual")
def compute_event_smdp_residual_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor | None = None,
    event_ids: torch.Tensor | None = None,
    event_values: torch.Tensor | None = None,
    intervention_influence: torch.Tensor | None = None,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    event_mix_lambda: float = 0.0,
    influence_temperature: float = 1.0,
    influence_beta: float = 0.0,
    influence_clip: float = 3.0,
    normalize_advantages: bool = True,
    loss_mask: torch.Tensor | None = None,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conservative V2 estimator: GAE plus a credit-conserving Event residual.

    ``event_mix_lambda=0`` is exactly action-level GAE, which makes the
    warm-up safe and directly testable.  The Event branch is enabled only by
    increasing the lambda after Influence Model calibration.  Returns target
    the original π0.5 critic, while ``V_E`` remains a separate sidecar.
    """
    if values is None:
        raise ValueError("event_smdp_residual requires the original PPO critic values")
    if not 0.0 <= event_mix_lambda <= 1.0:
        raise ValueError("event_mix_lambda must be in [0, 1]")
    # Local import avoids a module-level registry import cycle.
    from rlinf.algorithms.advantages import compute_gae_advantages_and_returns

    gae_advantages, _ = compute_gae_advantages_and_returns(
        rewards,
        gamma=gamma,
        gae_lambda=gae_lambda,
        values=values,
        dones=dones,
        normalize_advantages=False,
    )
    # Hard safety identity: during calibration lambda=0 must not even evaluate
    # the auxiliary Event tensors.  Besides saving compute, this prevents an
    # invalid sidecar value from leaking through IEEE ``0 * NaN`` into the
    # official GAE control.
    if event_mix_lambda == 0.0:
        returns = values[:-1] + gae_advantages
        if normalize_advantages:
            gae_advantages = safe_normalize(gae_advantages, loss_mask=loss_mask)
        return gae_advantages, returns
    if event_ids is None or event_values is None or intervention_influence is None:
        raise ValueError("nonzero event_smdp_residual requires event IDs/values and intervention influence")
    event_advantages, _ = event_smdp_credit(
        rewards,
        dones,
        event_ids,
        event_values,
        intervention_influence,
        gamma=gamma,
        influence_temperature=influence_temperature,
        influence_beta=influence_beta,
        influence_clip=influence_clip,
    )
    advantages = (1.0 - event_mix_lambda) * gae_advantages + event_mix_lambda * event_advantages
    # Do not let auxiliary Event redistribution retarget π0.5's original
    # critic.  The actor consumes the mixed advantage, while the PPO value
    # head continues to regress the independent GAE return; V_E has its own
    # remaining-event target in ``event_value.py``.
    returns = values[:-1] + gae_advantages
    if normalize_advantages:
        advantages = safe_normalize(advantages, loss_mask=loss_mask)
    return advantages, returns
