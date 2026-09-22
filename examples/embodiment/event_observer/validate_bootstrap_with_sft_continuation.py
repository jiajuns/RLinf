#!/usr/bin/env python3
"""Test whether branch bootstrap ranking predicts a fixed-SFT continuation.

This is a diagnostic, not PPO training.  From each live RoboTwin snapshot it
executes complete, sampled 5-step chunks, records
``R_branch + gamma * Vbar_E(endpoint)``, then restores the endpoint and lets
the *frozen* pi0.5 SFT policy run to task termination.  Thus it can separate a
bad Influence regressor from a branch target whose Event-Value bootstrap does
not rank actual remaining return.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.event_value import OnlineEventValueSidecar
from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
from rlinf.models.embodiment.openpi import get_model


def _model_observation(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None | list[str]]:
    # The standalone environment exposes the same real RGB/state fields as
    # the rollout worker.  `extra_view_images` is intentionally absent from
    # RoboTwin and must be explicit for OpenPI's adapter.
    return {**obs, "extra_view_images": None}


def _right_wrist(obs: dict[str, torch.Tensor]) -> torch.Tensor:
    wrist = obs["wrist_images"]
    if wrist.ndim == 5:  # [B,camera,H,W,3]
        return wrist[:, -1]
    if wrist.ndim == 4:
        return wrist
    raise ValueError("unexpected RoboTwin wrist image shape")


def _sidecar_value(sidecar: OnlineEventValueSidecar, history: list[dict[str, torch.Tensor]]) -> float:
    head = torch.cat([item["main_images"] for item in history], dim=0).unsqueeze(0).to("cuda")
    wrist = torch.cat([_right_wrist(item) for item in history], dim=0).unsqueeze(0).to("cuda")
    measured = torch.cat([item["measured_state16"] for item in history], dim=0).unsqueeze(0).to("cuda")
    return float(sidecar.infer_images(head, wrist, measured, use_target_value=True).values[0, -1].cpu())


def _endpoint_fingerprint(obs: dict[str, torch.Tensor]) -> dict[str, str]:
    """Hash each online sidecar input component, never simulator oracle fields."""
    def digest(value: torch.Tensor) -> str:
        return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]
    return {"head_rgb": digest(obs["main_images"]), "right_wrist_rgb": digest(_right_wrist(obs)),
            "measured_state16": digest(obs["measured_state16"])}


def _action_fingerprint(action: torch.Tensor) -> str:
    return hashlib.sha256(action.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def _sample_action(model: torch.nn.Module, obs: dict[str, torch.Tensor], seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    action, _ = model.predict_action_batch(_model_observation(obs), mode="eval", compute_values=False)
    return action.detach().cpu()


def _chunk(env: RoboTwinEnv, action: torch.Tensor) -> tuple[dict[str, torch.Tensor], float, bool]:
    observations, rewards, terminations, truncations, _ = env.chunk_step(action, auto_reset=False)
    done = bool(terminations.any() or truncations.any())
    return observations[-1], float(rewards.sum().cpu()), done


def _rank_metrics(bootstrap: np.ndarray, continuation: np.ndarray, tie_eps: float) -> dict[str, float]:
    concordant = discordant = eligible = 0
    for left in range(len(bootstrap)):
        for right in range(left + 1, len(bootstrap)):
            truth = continuation[left] - continuation[right]
            prediction = bootstrap[left] - bootstrap[right]
            if abs(float(truth)) <= tie_eps:
                continue
            eligible += 1
            if abs(float(prediction)) > tie_eps:
                if np.sign(truth) == np.sign(prediction):
                    concordant += 1
                else:
                    discordant += 1
    denominator = concordant + discordant
    return {
        "pairwise_accuracy_non_tied": concordant / max(eligible, 1),
        "concordant": concordant, "discordant": discordant, "eligible": eligible,
        "predicted_ties": eligible - denominator,
        "top1_regret": float(continuation.max() - continuation[np.argmax(bootstrap)]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", type=Path, required=True,
                        help="Resolved calibration YAML: gives the exact SFT and RoboTwin settings.")
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--rgb-student", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-chunks", default="5,15,25",
                        help="Comma-separated 5-step chunk indices; time-stratified without oracle state input.")
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--continuation-repeats", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tie-eps", type=float, default=1e-4)
    parser.add_argument("--duplicate-first-candidate", action="store_true",
                        help="Replace the last sampled candidate by candidate 0 to measure restore/input/value repeat noise.")
    args = parser.parse_args()
    if args.candidates < 2 or args.continuation_repeats < 1 or not 0 < args.gamma <= 1:
        raise ValueError("need >=2 candidates, >=1 repeat and gamma in (0,1]")
    if not torch.cuda.is_available():
        raise RuntimeError("this diagnostic intentionally requires CUDA for frozen pi0.5 inference")
    cfg = OmegaConf.load(args.resolved_config)
    cfg.env.train.auto_reset = False; cfg.env.train.ignore_terminations = False; cfg.env.train.total_num_envs = 1
    # Configured action chunk is the experimental decision time unit.  Reject
    # accidental 50-step model evaluation rather than silently changing gamma.
    if int(cfg.actor.model.num_action_chunks) != 5 or int(cfg.actor.model.openpi.action_chunk) != 5:
        raise ValueError("bootstrap diagnostic is defined for the validated 5-step action chunk")
    os.environ["ASSETS_PATH"] = str(cfg.env.train.assets_path)
    sidecar = OnlineEventValueSidecar.from_checkpoints(args.observer, args.rgb_student, device="cuda").eval()
    model = get_model(cfg.actor.model).to("cuda").eval()
    env = RoboTwinEnv(cfg.env.train, num_envs=1, seed_offset=0, total_num_processes=1, worker_info={}, record_metrics=True)
    target_chunks = sorted({int(value) for value in args.state_chunks.split(",") if value})
    if not target_chunks or target_chunks[0] < 0:
        raise ValueError("state chunks must be nonnegative")
    observation, _ = env.reset(); history = [observation]
    reports = []
    try:
        for current_chunk in range(max(target_chunks) + 1):
            if current_chunk in target_chunks:
                root = env.get_state(); root_history = list(history)
                candidate_actions = [_sample_action(model, observation, args.seed + 1009 * current_chunk + candidate) for candidate in range(args.candidates)]
                if args.duplicate_first_candidate:
                    candidate_actions[-1] = candidate_actions[0].clone()
                branch_rows = []
                try:
                    for candidate, action in enumerate(candidate_actions):
                        env.load_state(root)
                        endpoint, branch_reward, terminal = _chunk(env, action)
                        endpoint_history = root_history + [endpoint]
                        first_value = _sidecar_value(sidecar, endpoint_history)
                        second_value = _sidecar_value(sidecar, endpoint_history)
                        bootstrap = branch_reward if terminal else branch_reward + args.gamma * first_value
                        endpoint_state = env.get_state()
                        continuation_returns = []
                        for repeat in range(args.continuation_repeats):
                            env.load_state(endpoint_state)
                            continuation_obs = endpoint; continuation_reward = 0.0; discount = 1.0; done = terminal
                            # Same frozen SFT weights.  The repeat seed changes
                            # only policy sampling, quantifying continuation
                            # stochasticity instead of hiding it in one roll.
                            continuation_seed = args.seed + 100000 * current_chunk + 1000 * candidate + repeat
                            while not done and int(env.elapsed_steps[0]) < int(cfg.env.train.max_episode_steps):
                                follow = _sample_action(model, continuation_obs, continuation_seed + int(env.elapsed_steps[0]))
                                continuation_obs, reward, done = _chunk(env, follow)
                                continuation_reward += discount * reward; discount *= args.gamma
                            continuation_returns.append(continuation_reward)
                        branch_rows.append({
                            "candidate": candidate, "branch_reward": branch_reward, "terminal": terminal,
                            "bootstrap_target": bootstrap, "endpoint_value_first": first_value,
                            "endpoint_value_repeat_absdiff": abs(first_value - second_value),
                            "endpoint_input_fingerprint": _endpoint_fingerprint(endpoint),
                            "action_fingerprint": _action_fingerprint(action), "continuation_returns": continuation_returns,
                            "continuation_mean": float(np.mean(continuation_returns)), "continuation_std": float(np.std(continuation_returns)),
                        })
                finally:
                    env.load_state(root)
                bootstrap = np.asarray([row["bootstrap_target"] for row in branch_rows])
                continuation = np.asarray([row["continuation_mean"] for row in branch_rows])
                repeat_controls = []
                for left in range(len(branch_rows)):
                    for right in range(left + 1, len(branch_rows)):
                        if branch_rows[left]["action_fingerprint"] == branch_rows[right]["action_fingerprint"]:
                            repeat_controls.append({
                                "candidates": [left, right],
                                "endpoint_components_equal": {
                                    key: branch_rows[left]["endpoint_input_fingerprint"][key] == branch_rows[right]["endpoint_input_fingerprint"][key]
                                    for key in branch_rows[left]["endpoint_input_fingerprint"]
                                },
                                "bootstrap_absdiff": abs(branch_rows[left]["bootstrap_target"] - branch_rows[right]["bootstrap_target"]),
                                "value_absdiff": abs(branch_rows[left]["endpoint_value_first"] - branch_rows[right]["endpoint_value_first"]),
                            })
                reports.append({"chunk_index": current_chunk, "elapsed_control_steps": int(env.elapsed_steps[0]), "branches": branch_rows,
                                "repeat_controls": repeat_controls, "ranking": _rank_metrics(bootstrap, continuation, args.tie_eps)})
            if current_chunk == max(target_chunks):
                break
            action = _sample_action(model, observation, args.seed + current_chunk)
            observation, _, done = _chunk(env, action); history.append(observation)
            if done:
                break
    finally:
        env.offload()
    aggregate = {"states": len(reports), "pairwise_eligible": int(sum(row["ranking"]["eligible"] for row in reports)),
                 "pairwise_concordant": int(sum(row["ranking"]["concordant"] for row in reports)),
                 "pairwise_discordant": int(sum(row["ranking"]["discordant"] for row in reports))}
    aggregate["pairwise_accuracy_non_tied"] = aggregate["pairwise_concordant"] / max(aggregate["pairwise_eligible"], 1)
    result = {"protocol": {"policy": "frozen pi0.5 SFT", "action_chunk_control_steps": 5,
                             "branch_target": "R_branch + gamma * target_EventValue(endpoint)",
                             "continuation_target": "R_branch + gamma * fixed-SFT discounted continuation return",
                             "state_selection": "time-stratified; no oracle state input", "gamma": args.gamma},
              "aggregate": aggregate, "state_reports": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
