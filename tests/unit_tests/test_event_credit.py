import torch

from rlinf.algorithms.event_credit import event_smdp_credit


def test_event_smdp_uses_duration_discount_and_signed_interventional_allocation():
    """Credit preserves both discounted event value and action harm/benefit."""
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    dones = torch.tensor([[False], [False], [False], [True]])
    ids = torch.tensor([[0], [0], [1]])
    values = torch.tensor([[1.0], [1.0], [4.0], [9.0]])
    influence = torch.tensor([[0.0], [2.0], [0.0]])
    advantages, returns = event_smdp_credit(
        rewards, dones, ids, values, influence, gamma=0.9, influence_temperature=1.0
    )
    event_advantage = 1.0 + 0.9 * 2.0 + 0.9**2 * 4.0 - 1.0
    # The zero-influence step gets no causal credit; the beneficial second
    # action receives the event credit.  This is deliberately not softmax.
    torch.testing.assert_close(advantages[:2, 0], torch.tensor([0.0, 2 * abs(event_advantage)]))
    # Terminal events do not bootstrap from the (irrelevant) final value.
    torch.testing.assert_close(advantages[2, 0], torch.tensor(-1.0))
    torch.testing.assert_close(returns[:, 0], values[:-1, 0] + advantages[:, 0])


def test_event_smdp_negative_influence_is_punished_inside_positive_event():
    rewards = torch.tensor([[1.0], [1.0]])
    dones = torch.tensor([[False], [False], [True]])
    ids = torch.tensor([[0], [0]])
    values = torch.zeros(3, 1)
    influence = torch.tensor([[-1.0], [3.0]])
    advantages, _ = event_smdp_credit(rewards, dones, ids, values, influence, gamma=1.0)
    # A positive event advantage is split by |I| but retains the sign of I.
    torch.testing.assert_close(advantages[:, 0], torch.tensor([-1.0, 3.0]))


def test_event_ids_split_contiguous_runs_even_when_an_id_reappears():
    """A reused label starts a fresh event rather than merging distant runs."""
    rewards = torch.ones(3, 1)
    dones = torch.tensor([[False], [False], [False], [True]])
    ids = torch.tensor([[0], [1], [0]])
    values = torch.zeros(4, 1)
    influence = torch.zeros(3, 1)
    advantages, _ = event_smdp_credit(rewards, dones, ids, values, influence, gamma=0.9)
    torch.testing.assert_close(advantages, torch.ones_like(advantages))
