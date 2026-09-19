import torch

from rlinf.envs.sim.maniskill.event_oracle import (
    ALIGN,
    APPROACH,
    GRASP,
    LIFT,
    PLACE,
    TRANSPORT,
    classify_put_on_event,
)


def test_put_on_oracle_uses_current_grasp_height_distance_and_release():
    """Every oracle phase is driven by current privileged state, not history."""
    event_id, progress = classify_put_on_event(
        grasp_steps=torch.tensor([0, 1, 2, 2, 2, 0]),
        source_z=torch.tensor([0.0, 0.0, 0.0, 0.06, 0.06, 0.06]),
        initial_source_z=torch.zeros(6),
        gripper_source_distance=torch.tensor([0.01, 0.10, 0.10, 0.10, 0.10, 0.10]),
        source_target_distance=torch.tensor([0.4, 0.4, 0.4, 0.4, 0.04, 0.04]),
        source_on_target=torch.tensor([False, False, False, False, False, True]),
        success=torch.tensor([False, False, False, False, False, True]),
        grasp_confirmation_steps=2,
        lift_height=0.04,
        approach_distance=0.20,
        transport_distance=0.20,
        align_distance=0.08,
    )
    assert event_id.tolist() == [APPROACH, GRASP, LIFT, TRANSPORT, ALIGN, PLACE]
    assert torch.all((progress >= 0) & (progress <= 1))
    assert progress[-1].item() == 1.0
