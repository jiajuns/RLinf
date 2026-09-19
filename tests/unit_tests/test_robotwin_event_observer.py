"""Contracts for RobotWin Event Observer dataset materialization."""

import importlib.util
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
