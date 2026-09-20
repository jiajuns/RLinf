import torch

from rlinf.algorithms.event_intervention import (
    EventInfluenceModel,
    branch_event_returns,
    intervention_influence,
)


class FakeSnapshotEnvironment:
    def __init__(self):
        self.state = 7

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state


def test_branch_return_is_same_state_controlled_and_restores_live_simulator():
    """Every branch starts at one state and the caller's state survives."""
    env = FakeSnapshotEnvironment()
    starts = []

    def rollout(action):
        starts.append(env.state)
        env.state += action
        return torch.tensor(float(action)), torch.tensor(2.0), 2

    values = branch_event_returns(env, [1, 3], rollout, gamma=0.5)
    torch.testing.assert_close(values, torch.tensor([1.5, 3.5]))
    assert starts == [7, 7]
    assert env.state == 7
    torch.testing.assert_close(intervention_influence(values), torch.tensor([-1.0, 1.0]))


def test_influence_model_accepts_sparse_branch_supervision():
    model = EventInfluenceModel(event_representation_dim=3, action_dim=2, state_dim=1, hidden_dim=8)
    representation = torch.randn(2, 4, 3)
    actions = torch.randn(2, 4, 2)
    state = torch.randn(2, 4, 1)
    prediction = model(representation, actions, state)
    assert prediction.shape == (2, 4)
    loss = EventInfluenceModel.loss(prediction, torch.zeros_like(prediction), torch.tensor(
        [[True, False, True, False], [False, True, False, True]]
    ))
    assert torch.isfinite(loss)
