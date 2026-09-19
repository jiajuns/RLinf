"""Contracts for RobotWin Event Observer dataset materialization."""

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/embodiment/event_observer/prepare_robotwin_observer_dataset.py"
)
_SPEC = importlib.util.spec_from_file_location("robotwin_observer_dataset", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_discover_cameras_uses_recorded_hdf5_keys(tmp_path: Path) -> None:
    """An archive without front_camera must still expose wrist and head RGB."""
    path = tmp_path / "episode.hdf5"
    with h5py.File(path, "w") as handle:
        observation = handle.create_group("observation")
        for name in ("head_camera", "right_camera", "left_camera", "aux_camera"):
            observation.create_group(name).create_dataset("rgb", data=np.zeros((2,), dtype="u1"))
    with h5py.File(path, "r") as handle:
        assert _MODULE.discover_cameras(handle) == (
            "left_camera",
            "right_camera",
            "head_camera",
            "aux_camera",
        )


def test_replay_proprio_uses_only_current_and_past_samples() -> None:
    """Velocity is a causal finite difference over replay-readback poses."""
    state = np.zeros((3, 16), dtype=np.float32)
    state[:, 0] = (0.0, 0.2, 0.5)
    state[:, 7] = (0.1, 0.3, 0.6)
    state[:, 8] = (1.0, 1.4, 2.0)
    state[:, 15] = (0.8, 0.5, 0.2)
    proprio = _MODULE.replay_proprio(state, np.asarray((0.0, 0.2, 0.5)))
    np.testing.assert_allclose(proprio["ee_linear_velocity"][0], 0.0)
    np.testing.assert_allclose(proprio["ee_linear_velocity"][1, 0, 0], 1.0)
    np.testing.assert_allclose(proprio["ee_linear_velocity"][2, 0, 0], 1.0)
    np.testing.assert_allclose(proprio["ee_linear_velocity"][1, 1, 0], 2.0)
    np.testing.assert_allclose(proprio["gripper_opening_raw"][:, 0], (0.1, 0.3, 0.6))


def test_canonical_event_targets_are_exclusive_except_release() -> None:
    """Replay-only geometry produces the Event-SMDP state sequence."""
    steps = 6
    relations = np.zeros((steps, 12, 8, 8), dtype=np.float32)
    nodes = np.zeros((steps, 8, 24), dtype=np.float32)
    success = np.zeros(steps, dtype=bool)
    # moving object is node 0, target is 1, and left gripper is 2.
    relations[1:, 1, 0, 2] = 1.0  # held_by
    relations[2:, 10, 0, 0] = 1.0  # lifted
    nodes[2, 0, 2] = 0.1  # vertically dominant lift
    nodes[3, 0, 0] = 0.1  # horizontal transport
    relations[4, 0, 0, 1] = 1.0  # target alignment
    relations[5, 8, 0, 0] = 1.0  # released
    success[5] = True
    roster = json.dumps(
        [
            {"name": "moving", "type": "object"},
            {"name": "target", "type": "support"},
            {"name": "left_gripper", "type": "left_gripper"},
        ]
    )
    targets, state = _MODULE.canonical_event_targets(relations, nodes, success, roster)
    names = _MODULE.EVENT_NAMES
    assert [names[index] for index in state] == [
        "approach",
        "grasp",
        "lift",
        "transport",
        "align",
        "place",
    ]
    assert targets[5, names.index("release")] == 1.0


def test_general_roles_and_shared_primitives_do_not_require_task_id() -> None:
    """Role and primitive labels are derived from roster/relations, not task class."""
    roster = '[{"name":"moving","type":"object"},{"name":"target","type":"object"},{"name":"left","type":"left_gripper"}]'
    relations = np.zeros((2, len(_MODULE.RELATION_NAMES), 8, 8), dtype=np.float32)
    held = _MODULE.RELATION_NAMES.index("held_by")
    inside = _MODULE.RELATION_NAMES.index("inside")
    relations[:, held, 0, 2] = 1.0
    relations[1, inside, 0, 1] = 1.0
    geometric, state_change = _MODULE.shared_relation_targets(relations, roster)
    assert _MODULE.general_role_bindings(roster)[0]["role"] == "manipulated_object"
    assert geometric.shape == (2, len(_MODULE.GEOMETRIC_RELATION_PRIMITIVES))
    assert state_change.shape == (2, len(_MODULE.STATE_CHANGE_PRIMITIVES))
    assert geometric[:, _MODULE.GEOMETRIC_RELATION_PRIMITIVES.index("attached")].all()
    assert geometric[1, _MODULE.GEOMETRIC_RELATION_PRIMITIVES.index("inside_or_on_target")] == 1.0
