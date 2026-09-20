import torch

from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.event_credit import (
    compute_event_smdp_residual_advantages,
    event_smdp_credit,
)
from rlinf.algorithms.registry import calculate_adv_and_returns


def test_event_smdp_uses_duration_discount_and_conserves_event_credit():
    """Uniform Event-SMDP allocation sums to the discounted event advantage."""
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    dones = torch.tensor([[False], [False], [False], [True]])
    ids = torch.tensor([[0], [0], [1]])
    values = torch.tensor([[1.0], [1.0], [4.0], [9.0]])
    influence = torch.tensor([[0.0], [2.0], [0.0]])
    advantages, returns = event_smdp_credit(
        rewards, dones, ids, values, influence, gamma=0.9, influence_temperature=1.0
    )
    event_advantage = 1.0 + 0.9 * 2.0 + 0.9**2 * 4.0 - 1.0
    # No calibrated intervention means each action receives A_E / D.
    torch.testing.assert_close(advantages[:2, 0], torch.full((2,), event_advantage / 2))
    torch.testing.assert_close(advantages[:2, 0].sum(), torch.tensor(event_advantage))
    # Terminal events do not bootstrap from the (irrelevant) final value.
    torch.testing.assert_close(advantages[2, 0], torch.tensor(-1.0))
    torch.testing.assert_close(returns[:, 0], values[:-1, 0] + advantages[:, 0])


def test_event_smdp_influence_redistributes_without_changing_event_total():
    rewards = torch.tensor([[1.0], [1.0]])
    dones = torch.tensor([[False], [False], [True]])
    ids = torch.tensor([[0], [0]])
    values = torch.zeros(3, 1)
    influence = torch.tensor([[-1.0], [3.0]])
    advantages, _ = event_smdp_credit(
        rewards, dones, ids, values, influence, gamma=1.0, influence_beta=0.1
    )
    # Influence moves credit from one action to the other, but cannot decide
    # whether the whole event is rewarded or punished.
    torch.testing.assert_close(advantages[:, 0].sum(), torch.tensor(2.0))
    assert advantages[1, 0] > advantages[0, 0]


def test_event_ids_split_contiguous_runs_even_when_an_id_reappears():
    """A reused label starts a fresh event rather than merging distant runs."""
    rewards = torch.ones(3, 1)
    dones = torch.tensor([[False], [False], [False], [True]])
    ids = torch.tensor([[0], [1], [0]])
    values = torch.zeros(4, 1)
    influence = torch.zeros(3, 1)
    advantages, _ = event_smdp_credit(rewards, dones, ids, values, influence, gamma=0.9)
    torch.testing.assert_close(advantages, torch.ones_like(advantages))


def test_event_residual_lambda_zero_is_exactly_gae():
    rewards = torch.tensor([[0.2], [0.3]])
    dones = torch.tensor([[False], [False], [True]])
    values = torch.tensor([[0.1], [0.2], [0.0]])
    ids = torch.zeros(2, 1, dtype=torch.long)
    influence = torch.tensor([[8.0], [-8.0]])
    expected, expected_returns = compute_gae_advantages_and_returns(
        rewards, dones=dones, values=values, gamma=0.9, gae_lambda=0.95, normalize_advantages=False
    )
    actual, actual_returns = compute_event_smdp_residual_advantages(
        rewards,
        dones,
        values,
        ids,
        values,
        influence,
        gamma=0.9,
        gae_lambda=0.95,
        event_mix_lambda=0.0,
        influence_beta=1.0,
        normalize_advantages=False,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_returns, expected_returns)


def test_chunk_level_residual_keeps_branch_and_credit_granularity_aligned():
    """A [C,B,K] sidecar is reduced to one counterfactual per policy chunk."""
    chunks, batch, action_chunk = 2, 1, 3
    rewards = torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0]]])
    dones = torch.zeros(chunks + 1, batch, action_chunk, dtype=torch.bool)
    dones[-1] = True
    values = torch.zeros(chunks + 1, batch, 1)
    ids = torch.zeros(chunks, batch, action_chunk, dtype=torch.long)
    event_values = torch.zeros(chunks + 1, batch, action_chunk)
    influence = torch.tensor([[[10.0, 0.0, 0.0]], [[-10.0, 0.0, 0.0]]])
    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="event_smdp_residual",
        rewards=rewards,
        dones=dones,
        values=values,
        event_ids=ids,
        event_values=event_values,
        intervention_influence=influence,
        reward_type="chunk_level",
        logprob_type="chunk_level",
        gamma=1.0,
        gae_lambda=1.0,
        event_mix_lambda=0.0,
        normalize_advantages=False,
    )
    assert result["advantages"].shape == (chunks, batch, 1)
    # lambda=0 must stay a valid GAE baseline even when branch scores exist.
    torch.testing.assert_close(result["advantages"][:, 0, 0], torch.tensor([1.0, 0.0]))
