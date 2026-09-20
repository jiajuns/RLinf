import torch

from rlinf.algorithms.event_value import OnlineEventValueSidecar
from rlinf.models.embodiment.event_observer import EventObserver, EventValueCritic


def test_frozen_observer_infers_local_event_ids_and_online_value_loss():
    observer = EventObserver(5, 3, 4, hidden_dim=8, num_geometric_primitives=2, num_state_change_primitives=1)
    sidecar = OnlineEventValueSidecar(observer, EventValueCritic(8, hidden_dim=8))
    assert not any(parameter.requires_grad for parameter in sidecar.observer.parameters())
    features = torch.randn(2, 4, 5)
    output = sidecar.infer(features, boundary_threshold=0.5)
    assert output.values.shape == (2, 4)
    assert torch.equal(output.event_ids[:, 0], torch.zeros(2, dtype=torch.long))
    representation = torch.randn(2, 5, 8)
    rewards = torch.zeros(4, 2)
    rewards[-1] = 1
    dones = torch.zeros(5, 2, dtype=torch.bool)
    dones[-1] = True
    event_ids = torch.tensor([[0, 0], [0, 0], [1, 1], [1, 1]])
    loss = sidecar.smdp_value_loss(representation, rewards, dones, event_ids, gamma=0.99)
    assert torch.isfinite(loss) and loss.item() >= 0
