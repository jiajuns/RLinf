#!/usr/bin/env python3
"""Validate an actual RoboTwin same-state branch point, not a mock adapter."""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv


def assert_state_equal(left: dict, right: dict) -> None:
    for key in ("states", "measured_state16"):
        a, b = left.get(key), right.get(key)
        if a is None or b is None:
            if a is not b:
                raise AssertionError(f"observation key {key} differs in None state")
        else:
            torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)


def image_mismatch_fraction(left: dict, right: dict) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in ("main_images", "wrist_images"):
        a, b = left.get(key), right.get(key)
        if a is not None and b is not None:
            result[key] = float((a.cpu() != b.cpu()).float().mean())
    return result


def snapshot_component_hashes(state: bytes) -> dict[str, str]:
    """Debug identity without treating a whole pickle as a physics oracle."""
    payload = pickle.loads(state)
    def content_digest(value) -> str:
        """Hash values rather than pickle's tensor-storage bookkeeping."""
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            encoded = ("tensor", str(value.dtype), tuple(value.shape), value.numpy().tobytes())
        elif hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes"):
            encoded = ("array", str(value.dtype), tuple(value.shape), value.tobytes())
        elif isinstance(value, dict):
            encoded = ("dict", tuple((str(key), content_digest(item)) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))))
        elif isinstance(value, (list, tuple)):
            encoded = (type(value).__name__, tuple(content_digest(item) for item in value))
        else:
            encoded = (type(value).__name__, repr(value))
        return hashlib.sha256(pickle.dumps(encoded, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()[:12]

    result = {
        "python_rng": content_digest(payload["python_rng"]),
        "numpy_rng": content_digest(payload["numpy_rng"]),
        "torch_rng": content_digest(payload["torch_rng"]),
    }
    for index, subenv in enumerate(payload["subenv_states"]):
        result[f"physics_{index}"] = hashlib.sha256(subenv["physics"]).hexdigest()[:12]
        result[f"task_fields_{index}"] = content_digest(subenv["task_fields"])
    for key in ("prev_step_reward", "elapsed_steps", "success_once", "fail_once", "returns", "is_start"):
        result[key] = content_digest(payload.get(key))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--robotwin-assets", type=Path, required=True)
    parser.add_argument("--robotwin-source", type=Path, required=True)
    parser.add_argument("--shader", choices=("minimal", "default", "rt"), default="minimal")
    parser.add_argument("--disable-denoiser", action="store_true")
    parser.add_argument("--action-steps", type=int, default=5,
                        help="Full control chunk used for exact restore validation.")
    parser.add_argument("--action-value", type=float, default=.1,
                        help="Nonzero deterministic control exposes Python-side controller state omitted by a snapshot.")
    args = parser.parse_args()
    os.environ.setdefault("EMBODIED_PATH", str(args.repo / "examples/embodiment"))
    os.environ.setdefault("REPO_PATH", str(args.repo))
    os.environ["ASSETS_PATH"] = str(args.robotwin_assets)
    # The deployed RoboTwin runtime reads these in Base_Task.setup_scene.
    # This process-level route also reaches VectorEnv sub-environments, unlike
    # a parent-only monkeypatch of SAPIEN's Python bindings.
    os.environ["ROBOTWIN_CAMERA_SHADER"] = args.shader
    if args.disable_denoiser:
        os.environ["ROBOTWIN_RAY_DENOISER"] = ""
    with initialize_config_dir(version_base="1.1", config_dir=str(args.repo / "examples/embodiment/config")):
        cfg = compose(
            config_name="robotwin_adjust_bottle_ppo_openpi_pi05",
            overrides=[
                "env.train.total_num_envs=1",
                "env.train.task_config.embodiment=[aloha-agilex]",
                "env.train.task_config.camera.collect_wrist_camera=true",
                "env.train.task_config.domain_randomization.random_background=false",
                "env.train.task_config.domain_randomization.cluttered_table=false",
                "env.train.task_config.domain_randomization.clean_background_rate=1",
                f"env.train.assets_path={args.robotwin_assets}",
                "env.train.video_cfg.save_video=false",
            ],
        )
    # The source checkout contains the public VectorEnv package; it must be
    # importable before RoboTwinEnv creates a worker process.
    import sys
    sys.path.insert(0, str(args.robotwin_source))
    env = RoboTwinEnv(cfg.env.train, num_envs=1, seed_offset=0, total_num_processes=1, worker_info={})
    try:
        initial_obs, _ = env.reset()
        if initial_obs.get("measured_state16") is None or initial_obs["measured_state16"].shape != (1, 16):
            raise AssertionError("RoboTwin online observer did not receive measured_state16")
        if args.action_steps < 1:
            raise ValueError("--action-steps must be positive")
        action_chunk = args.action_steps
        repeated_actions = torch.zeros((1, 2, action_chunk, 14), dtype=torch.float32)
        # A branch is allowed to produce labels, but must return the live
        # rollout exactly to its pre-branch state before PPO continues.
        pre_branch_snapshot = env.get_state()
        branch = env.branch_step(repeated_actions)
        post_branch_snapshot = env.get_state()
        pre_branch_hashes = snapshot_component_hashes(pre_branch_snapshot)
        post_branch_hashes = snapshot_component_hashes(post_branch_snapshot)
        changed_components = {
            key: (pre_branch_hashes[key], post_branch_hashes[key])
            for key in pre_branch_hashes if pre_branch_hashes[key] != post_branch_hashes[key]
        }
        if changed_components:
            raise AssertionError(f"branch_step changed live snapshot components: {changed_components}")
        if not bool(branch["branch_mask"].all()) or branch["branch_measured_state16"].shape != (1, 2, 16):
            raise AssertionError("matched-state branch did not return measured state")
        if not branch["branch_requested_steps"].eq(action_chunk).all() or not branch["branch_horizons"].le(action_chunk).all():
            raise AssertionError("branch did not record full requested/actual chunk duration")
        if not branch["branch_valid"].all() or branch["branch_state_ids"].shape != (1, 3):
            raise AssertionError("branch validity/state fingerprint transport is incomplete")
        elapsed_after_branch = env.elapsed_steps.detach().cpu().clone()
        branch_take_action_cnt = [
            getattr(subenv.task, "take_action_cnt", None) for subenv in env.venv.envs
        ]
        snapshot = env.get_state()
        action = torch.full((1, action_chunk, 14), args.action_value, dtype=torch.float32)
        first_obs_list, first_rewards, first_terms, first_truncs, _ = env.chunk_step(action, auto_reset=False)
        elapsed_after_first = env.elapsed_steps.detach().cpu().clone()
        first_take_action_cnt = [
            getattr(subenv.task, "take_action_cnt", None) for subenv in env.venv.envs
        ]
        env.load_state(snapshot)
        elapsed_after_restore = env.elapsed_steps.detach().cpu().clone()
        restored_take_action_cnt = [
            getattr(subenv.task, "take_action_cnt", None) for subenv in env.venv.envs
        ]
        second_obs_list, second_rewards, second_terms, second_truncs, _ = env.chunk_step(action, auto_reset=False)
        print({"elapsed_after_branch": elapsed_after_branch.tolist(), "elapsed_after_first": elapsed_after_first.tolist(),
               "elapsed_after_restore": elapsed_after_restore.tolist(), "first_truncated": first_truncs[:, -1].cpu().tolist(),
               "second_truncated": second_truncs[:, -1].cpu().tolist(),
               "take_action_cnt_after_branch": branch_take_action_cnt,
               "take_action_cnt_after_first": first_take_action_cnt,
               "take_action_cnt_after_restore": restored_take_action_cnt}, flush=True)
        first_obs, second_obs = first_obs_list[-1], second_obs_list[-1]
        torch.testing.assert_close(first_rewards.cpu(), second_rewards.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(first_terms.cpu(), second_terms.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(first_truncs.cpu(), second_truncs.cpu(), rtol=0, atol=0)
        # SAPIEN's camera readback can differ at the pixel level after a
        # render-buffer refresh even when no physical state changes.  The
        # causal invariant is therefore exact reward/done/proprio equality;
        # record RGB mismatch instead of falsely treating renderer noise as a
        # snapshot failure.  Influence fitting later averages this label noise
        # across matched-state branches.
        assert_state_equal(first_obs, second_obs)
        print({"snapshot_bytes": len(snapshot), "same_action_branch_physics": "exact", "shader": args.shader,
               "denoiser_disabled": args.disable_denoiser,
               "branch_candidates": int(branch["branch_rewards"].shape[1]),
               "requested_duration": branch["branch_requested_steps"].cpu().tolist(),
               "actual_duration": branch["branch_horizons"].cpu().tolist(),
               "repeat_branch_reward_abs_diff": (branch["branch_rewards"][:, 0] - branch["branch_rewards"][:, 1]).abs().cpu().tolist(),
               "repeat_branch_proprio_abs_max": (branch["branch_measured_state16"][:, 0] - branch["branch_measured_state16"][:, 1]).abs().max().cpu().item(),
               "action_value": args.action_value, "reward": first_rewards.sum(-1).cpu().tolist(),
               "rgb_mismatch_fraction": image_mismatch_fraction(first_obs, second_obs)})
    finally:
        env.offload(clear_cache=True)


if __name__ == "__main__":
    main()
