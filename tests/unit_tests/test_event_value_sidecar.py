import torch

from rlinf.algorithms.event_value import OnlineEventValueSidecar, infer_event_sidecar_rollout
from rlinf.models.embodiment.event_observer import EventObserver, EventValueCritic, RGBRoleFeatureStudent


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


def test_event_sidecar_can_use_a_frozen_rgb_student_online():
    observer = EventObserver(30, 3, 4, hidden_dim=8, num_geometric_primitives=2, num_state_change_primitives=1)
    sidecar = OnlineEventValueSidecar(observer, EventValueCritic(8, hidden_dim=8),
                                      RGBRoleFeatureStudent(30, hidden_dim=8, image_size=(32, 32)))
    output = sidecar.infer_images(
        torch.randint(0, 255, (1, 2, 40, 40, 3), dtype=torch.uint8),
        torch.randint(0, 255, (1, 2, 40, 40, 3), dtype=torch.uint8),
    )
    assert output.values.shape == (1, 2)


def test_event_sidecar_rollout_expands_chunk_boundary_rgb_without_cache_lookup():
    observer = EventObserver(30, 3, 4, hidden_dim=8, num_geometric_primitives=2, num_state_change_primitives=1)
    sidecar = OnlineEventValueSidecar(
        observer,
        EventValueCritic(8, hidden_dim=8),
        RGBRoleFeatureStudent(30, hidden_dim=8, image_size=(32, 32)),
    )
    # RLinf embodied trajectories are time-major.  Two wrist streams test the
    # explicit right-wrist selection used by the task-matched student.
    current = {
        "main_images": torch.randint(0, 255, (3, 2, 40, 40, 3), dtype=torch.uint8),
        "wrist_images": torch.randint(0, 255, (3, 2, 2, 40, 40, 3), dtype=torch.uint8),
    }
    successor = {key: value.clone() for key, value in current.items()}
    rollout = infer_event_sidecar_rollout(
        sidecar, current, successor, num_action_chunks=4, boundary_threshold=0.5
    )
    assert rollout.event_ids.shape == (3, 2, 4)
    assert rollout.event_values.shape == (4, 2, 4)
    assert rollout.action_representation.shape == (2, 13, 8)
    # Each action in one policy chunk shares the causally available boundary
    # event state; the final representation is a bootstrap only.
    assert torch.equal(rollout.event_ids[:, :, 0], rollout.event_ids[:, :, -1])
