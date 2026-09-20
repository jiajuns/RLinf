import importlib.util
from pathlib import Path

import numpy as np


MODULE = Path(__file__).parents[2] / "examples/embodiment/event_observer/audit_adjust_bottle_sam_tracks.py"
SPEC = importlib.util.spec_from_file_location("sam_audit", MODULE)
sam_audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(sam_audit)


def test_sam_audit_accepts_healthy_head_track_and_rejects_collapsed_track(tmp_path):
    healthy = np.zeros((2, 4, 6), np.float32)
    healthy[1, :, :] = [0.5, 0.5, 0.2, 0.3, 0.9, 1.0]
    good_path = tmp_path / "good.npz"
    np.savez(good_path, format=np.asarray("event_sam31_track_v1"), cameras=np.asarray(["right_camera", "head_camera"]),
             tracks=healthy, episode_sha256=np.asarray("x"), checkpoint_sha256=np.asarray("y"))
    assert sam_audit.track_health(good_path, min_visible_fraction=.5, min_box_area=.002)[0]
    healthy[1, :, 2:4] = 1e-4
    bad_path = tmp_path / "bad.npz"
    np.savez(bad_path, format=np.asarray("event_sam31_track_v1"), cameras=np.asarray(["right_camera", "head_camera"]),
             tracks=healthy, episode_sha256=np.asarray("x"), checkpoint_sha256=np.asarray("y"))
    ok, info = sam_audit.track_health(bad_path, min_visible_fraction=.5, min_box_area=.002)
    assert not ok and info["reason"] == "collapsed_head_box"
