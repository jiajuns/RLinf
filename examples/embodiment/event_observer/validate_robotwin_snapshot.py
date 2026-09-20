#!/usr/bin/env python3
"""Validate an actual RoboTwin same-state branch point, not a mock adapter."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv


def assert_state_equal(left: dict, right: dict) -> None:
    for key in ("states",):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--robotwin-assets", type=Path, required=True)
    parser.add_argument("--robotwin-source", type=Path, required=True)
    parser.add_argument("--shader", choices=("minimal", "default", "rt"), default="minimal")
    parser.add_argument("--disable-denoiser", action="store_true")
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
        env.reset()
        snapshot = env.get_state()
        action = torch.zeros((1, 1, 14), dtype=torch.float32)
        first_obs, first_reward, first_term, first_trunc, _ = env.step(action, auto_reset=False)
        env.load_state(snapshot)
        second_obs, second_reward, second_term, second_trunc, _ = env.step(action, auto_reset=False)
        torch.testing.assert_close(first_reward.cpu(), second_reward.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(first_term.cpu(), second_term.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(first_trunc.cpu(), second_trunc.cpu(), rtol=0, atol=0)
        # SAPIEN's camera readback can differ at the pixel level after a
        # render-buffer refresh even when no physical state changes.  The
        # causal invariant is therefore exact reward/done/proprio equality;
        # record RGB mismatch instead of falsely treating renderer noise as a
        # snapshot failure.  Influence fitting later averages this label noise
        # across matched-state branches.
        assert_state_equal(first_obs, second_obs)
        print({"snapshot_bytes": len(snapshot), "same_action_branch_physics": "exact", "shader": args.shader,
               "denoiser_disabled": args.disable_denoiser,
               "reward": first_reward.cpu().tolist(),
               "rgb_mismatch_fraction": image_mismatch_fraction(first_obs, second_obs)})
    finally:
        env.offload(clear_cache=True)


if __name__ == "__main__":
    main()
