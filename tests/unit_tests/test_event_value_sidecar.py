import torch
import pytest

from rlinf.algorithms.event_value import (
    OnlineEventValueSidecar,
    event_boundary_value_targets,
    infer_branch_future_event_values,
    infer_event_sidecar_rollout,
)
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
    terminations = torch.zeros(5, 2, dtype=torch.bool)
    terminations[-1] = True
    truncations = torch.zeros_like(terminations)
    event_ids = torch.tensor([[0, 0], [0, 0], [1, 1], [1, 1]])
    loss = sidecar.smdp_value_loss(
        representation, rewards, terminations, truncations, event_ids, gamma=0.99
    )
    assert torch.isfinite(loss) and loss.item() >= 0


def test_event_value_targets_correct_each_internal_chunk_not_event_credit_average():
    rewards = torch.tensor([[1.0], [2.0]])
    terminations = torch.zeros(3, 1, dtype=torch.bool)
    truncations = torch.zeros_like(terminations)
    ids = torch.zeros(2, 1, dtype=torch.long)
    target_values = torch.tensor([[0.0], [0.0], [3.0]])
    targets, valid = event_boundary_value_targets(
        rewards, terminations, truncations, ids, target_values,
        gamma=0.5, bootstrap_on_truncation=False,
    )
    assert valid.all()
    # t=0 retains both rewards; t=1 sees only its remaining reward.  These
    # targets must not be a uniform event advantage copied to every chunk.
    torch.testing.assert_close(targets[:, 0], torch.tensor([2.75, 3.5]))
    truncations[-1] = True
    truncated_targets, _ = event_boundary_value_targets(
        rewards, terminations, truncations, ids, target_values,
        gamma=0.5, bootstrap_on_truncation=False,
    )
    torch.testing.assert_close(truncated_targets[:, 0], torch.tensor([2.0, 2.0]))


def test_event_value_target_regression_terminal_padding_and_truncation_contract():
    """Required two-step regression: [0,1], terminal, gamma=1 -> [1,1]."""
    rewards = torch.tensor([[0.0], [1.0], [99.0]])
    terminations = torch.tensor([[False], [False], [True], [False]])
    truncations = torch.zeros_like(terminations)
    ids = torch.tensor([[0], [0], [-1]])
    target_values = torch.tensor([[0.0], [0.0], [123.0], [456.0]])
    valid = torch.tensor([[True], [True], [False]])
    targets, target_mask = event_boundary_value_targets(
        rewards, terminations, truncations, ids, target_values,
        gamma=1.0, bootstrap_on_truncation=False, valid_mask=valid,
    )
    torch.testing.assert_close(targets[:2, 0], torch.ones(2))
    assert not target_mask[2, 0]  # padding cannot enter a V_E loss.

    # A true terminal never bootstraps, while a sampling truncation follows
    # the explicit switch rather than being silently treated as termination.
    terminations[2] = False
    truncations[2] = True
    no_bootstrap, _ = event_boundary_value_targets(
        rewards[:2], terminations[:3], truncations[:3], ids[:2], target_values[:3],
        gamma=1.0, bootstrap_on_truncation=False,
    )
    with_bootstrap, _ = event_boundary_value_targets(
        rewards[:2], terminations[:3], truncations[:3], ids[:2], target_values[:3],
        gamma=1.0, bootstrap_on_truncation=True,
    )
    torch.testing.assert_close(no_bootstrap[:, 0], torch.ones(2))
    torch.testing.assert_close(with_bootstrap[:, 0], torch.full((2,), 124.0))


def test_event_sidecar_can_use_a_frozen_rgb_student_online():
    observer = EventObserver(30, 3, 4, hidden_dim=8, num_geometric_primitives=2, num_state_change_primitives=1)
    sidecar = OnlineEventValueSidecar(observer, EventValueCritic(8, hidden_dim=8),
                                      RGBRoleFeatureStudent(30, hidden_dim=8, image_size=(32, 32)))
    output = sidecar.infer_images(
        torch.randint(0, 255, (1, 2, 40, 40, 3), dtype=torch.uint8),
        torch.randint(0, 255, (1, 2, 40, 40, 3), dtype=torch.uint8),
    )
    assert output.values.shape == (1, 2)


def test_deployable_sidecar_requires_matching_chunk_input_contract(tmp_path):
    """Offline checkpoints must declare the same online delta/mount inputs."""
    observer = EventObserver(5, 3, 4, num_geometric_primitives=2, num_state_change_primitives=1)
    value = EventValueCritic(256)
    student = RGBRoleFeatureStudent(5)
    contract = {
        "version": 1, "proprio_time_delta": 50.0, "mount_token": 2,
        "control_step_stride": 50, "proprio_derivative_unit": "per_control_step",
    }
    observer_path = tmp_path / "observer.pt"
    student_path = tmp_path / "student.pt"
    torch.save(
        {
            "observer": observer.state_dict(), "event_value": value.state_dict(),
            "feature_dim": 5, "posterior_dim": 3, "state_dim": 4,
            "geometric_dim": 2, "state_change_dim": 1,
            "online_input_contract": contract,
        }, observer_path,
    )
    torch.save(
        {"rgb_student": student.state_dict(), "feature_dim": 5, "online_input_contract": contract},
        student_path,
    )
    sidecar = OnlineEventValueSidecar.from_checkpoints(
        observer_path, student_path, proprio_time_delta=50.0, online_mount_token=2
    )
    assert sidecar.proprio_time_delta == 50.0 and sidecar.online_mount_token == 2
    with pytest.raises(ValueError, match="proprio_time_delta"):
        OnlineEventValueSidecar.from_checkpoints(
            observer_path, student_path, proprio_time_delta=1.0, online_mount_token=2
        )


def test_rgb_student_consumes_measured_proprioception_causally():
    student = RGBRoleFeatureStudent(30, hidden_dim=8, image_size=(32, 32))
    state = torch.zeros(1, 3, 16)
    state[0, :, 7] = torch.tensor([0.0, 0.5, 1.0])
    features = student.measured_proprio_features(state, time_delta=0.1)
    assert features.shape == (1, 3, 12)
    torch.testing.assert_close(features[0, 1, 2], torch.tensor(5.0))


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


def test_branch_future_values_keep_the_causal_rgb_prefix():
    observer = EventObserver(30, 3, 4, hidden_dim=8, num_geometric_primitives=2, num_state_change_primitives=1)
    sidecar = OnlineEventValueSidecar(
        observer,
        EventValueCritic(8, hidden_dim=8),
        RGBRoleFeatureStudent(30, hidden_dim=8, image_size=(32, 32)),
    )
    current = {
        "main_images": torch.randint(0, 255, (2, 1, 40, 40, 3), dtype=torch.uint8),
        "wrist_images": torch.randint(0, 255, (2, 1, 2, 40, 40, 3), dtype=torch.uint8),
    }
    branch_head = torch.randint(0, 255, (2, 1, 3, 40, 40, 3), dtype=torch.uint8)
    branch_wrist = torch.randint(0, 255, (2, 1, 3, 2, 40, 40, 3), dtype=torch.uint8)
    future_values = infer_branch_future_event_values(sidecar, current, branch_head, branch_wrist)
    assert future_values.shape == (2, 1, 3)
    assert torch.isfinite(future_values).all()
