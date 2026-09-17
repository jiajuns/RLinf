import torch

from rlinf.algorithms.event_intervention import (
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
