"""Privileged, semantically grounded event labels for ManiSkill ablations.

These labels are training-time supervision only.  The learned Role-Graph Event
Observer replaces this module at visual-only inference time.
"""

import torch


APPROACH = 0
GRASP = 1
LIFT = 2
TRANSPORT = 3
ALIGN = 4
PLACE = 5


def classify_put_on_event(
    *,
    grasp_steps: torch.Tensor,
    source_z: torch.Tensor,
    initial_source_z: torch.Tensor,
    gripper_source_distance: torch.Tensor,
    source_target_distance: torch.Tensor,
    source_on_target: torch.Tensor,
    success: torch.Tensor,
    grasp_confirmation_steps: int,
    lift_height: float,
    approach_distance: float,
    transport_distance: float,
    align_distance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [B] semantic event ids and within-event progress in [0, 1].

    ``grasp_steps`` must be the *current* raw simulator counter, not an
    episode-level boolean statistic.  The precedence below makes labels
    mutually exclusive and gives place only after a successful release.
    """
    if grasp_confirmation_steps < 1:
        raise ValueError("grasp_confirmation_steps must be positive")
    if lift_height <= 0 or approach_distance <= 0 or align_distance <= 0:
        raise ValueError("event distance/height thresholds must be positive")
    if transport_distance <= align_distance:
        raise ValueError("transport_distance must exceed align_distance")

    tensors = (
        grasp_steps,
        source_z,
        initial_source_z,
        gripper_source_distance,
        source_target_distance,
        source_on_target,
        success,
    )
    shape = grasp_steps.shape
    if any(tensor.shape != shape for tensor in tensors):
        raise ValueError("all oracle event inputs must have identical [B] shapes")

    current_grasped = grasp_steps > 0
    confirmed = grasp_steps >= grasp_confirmation_steps
    lift_delta = (source_z - initial_source_z).clamp_min(0.0)
    lifted = confirmed & (lift_delta >= lift_height)
    aligned = lifted & (source_target_distance <= align_distance)
    placed = success.to(torch.bool) & source_on_target.to(torch.bool) & ~current_grasped

    event_id = torch.full_like(grasp_steps, APPROACH, dtype=torch.long)
    event_id[current_grasped & ~confirmed] = GRASP
    event_id[confirmed & ~lifted] = LIFT
    event_id[lifted & ~aligned] = TRANSPORT
    event_id[aligned] = ALIGN
    event_id[placed] = PLACE

    progress = torch.zeros_like(source_z, dtype=torch.float32)
    progress[event_id == APPROACH] = 1.0 - (
        gripper_source_distance[event_id == APPROACH] / approach_distance
    ).clamp(0.0, 1.0)
    progress[event_id == GRASP] = (
        grasp_steps[event_id == GRASP].float() / grasp_confirmation_steps
    ).clamp(0.0, 1.0)
    progress[event_id == LIFT] = (
        lift_delta[event_id == LIFT] / lift_height
    ).clamp(0.0, 1.0)
    transport_span = transport_distance - align_distance
    progress[event_id == TRANSPORT] = 1.0 - (
        (source_target_distance[event_id == TRANSPORT] - align_distance)
        / transport_span
    ).clamp(0.0, 1.0)
    progress[event_id == ALIGN] = 1.0 - (
        source_target_distance[event_id == ALIGN] / align_distance
    ).clamp(0.0, 1.0)
    progress[event_id == PLACE] = 1.0
    return event_id, progress.clamp(0.0, 1.0)
