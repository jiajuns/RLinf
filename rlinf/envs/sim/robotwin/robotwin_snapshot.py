"""Strict same-state snapshots for RoboTwin/SAPIEN short interventions.

RoboTwin's public vector wrapper does not expose a snapshot API, but every
live task owns a SAPIEN PhysX CPU system whose ``pack()/unpack()`` byte state
contains all scene actors, articulations and velocities.  This adapter adds
the small Python-side episode bookkeeping required for reward-equivalent
short branches.  It is intentionally separate from reset: a reset is never a
valid causal intervention.
"""
from __future__ import annotations

import copy
import pickle
import random
from typing import Any

import numpy as np
import torch


_TASK_FIELDS = (
    # RoboTwin exposes two action counters in different task paths.
    # ``run_steps`` is used by dense-reward tasks, while
    # ``gen_sparse_reward_data`` uses ``take_action_cnt`` to enforce its
    # horizon.  Omitting the latter makes a candidate branch leak its action
    # count into the next candidate even though PhysX was restored.
    "run_steps", "take_action_cnt", "reward_step", "eval_success", "stage_success_tag",
    "plan_success", "instruction", "info",
)


def _snapshot_subenv(subenv: Any) -> dict[str, Any]:
    task = subenv.task
    physics = task.scene.get_physx_system()
    fields = {name: copy.deepcopy(getattr(task, name)) for name in _TASK_FIELDS if hasattr(task, name)}
    return {"physics": physics.pack(), "task_fields": fields, "instruction": copy.deepcopy(subenv.instruction)}


def _restore_subenv(subenv: Any, state: dict[str, Any]) -> None:
    task = subenv.task
    task.scene.get_physx_system().unpack(state["physics"])
    for name, value in state["task_fields"].items():
        setattr(task, name, copy.deepcopy(value))
    subenv.instruction = copy.deepcopy(state["instruction"])
    # Rendering is not part of PhysX state. Refreshing it after unpack only
    # updates camera buffers and does not step physics or mutate task state.
    refresh = getattr(task, "_update_render", None)
    if callable(refresh):
        refresh()


def snapshot_robotwin_env(env: Any) -> bytes:
    """Serialize a complete RLinf ``RoboTwinEnv`` branch point."""
    venv = env.venv
    global_lock = getattr(venv, "global_lock", None)
    context = global_lock if global_lock is not None else _NullContext()
    with context:
        subenv_states = []
        for subenv in venv.envs:
            with subenv.lock:
                subenv_states.append(_snapshot_subenv(subenv))
        payload = {
            "format": "rlinf_robotwin_snapshot_v1",
            "subenv_states": subenv_states,
            # ``.cpu()`` aliases storage when RoboTwin runs on CPU. Branches
            # mutate these tensors in-place, so every bookkeeping tensor must
            # be cloned or a supposedly restored branch leaks elapsed/done
            # state into the live trajectory.
            "prev_step_reward": env.prev_step_reward.detach().cpu().clone(),
            "elapsed_steps": getattr(env, "_elapsed_steps", None).detach().cpu().clone()
            if hasattr(env, "_elapsed_steps") else None,
            "success_once": getattr(env, "success_once", None).detach().cpu().clone()
            if hasattr(env, "success_once") else None,
            "fail_once": getattr(env, "fail_once", None).detach().cpu().clone()
            if hasattr(env, "fail_once") else None,
            "returns": getattr(env, "returns", None).detach().cpu().clone()
            if hasattr(env, "returns") else None,
            "is_start": env.is_start,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
        }
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def restore_robotwin_env(env: Any, state_buffer: bytes) -> None:
    """Restore a :func:`snapshot_robotwin_env` byte state exactly."""
    payload = pickle.loads(state_buffer)
    if payload.get("format") != "rlinf_robotwin_snapshot_v1":
        raise ValueError("unrecognized RoboTwin snapshot format")
    if len(payload["subenv_states"]) != len(env.venv.envs):
        raise ValueError("snapshot env count does not match live RoboTwinEnv")
    venv = env.venv
    global_lock = getattr(venv, "global_lock", None)
    context = global_lock if global_lock is not None else _NullContext()
    with context:
        for subenv, saved in zip(venv.envs, payload["subenv_states"], strict=True):
            with subenv.lock:
                _restore_subenv(subenv, saved)
        for name, value in (("prev_step_reward", payload["prev_step_reward"]),
                            ("_elapsed_steps", payload["elapsed_steps"]),
            ("success_once", payload["success_once"]),
            ("fail_once", payload["fail_once"]),
            ("returns", payload["returns"])):
            if value is not None:
                # Do not alias the serialized payload on CPU: subsequent
                # in-place environment bookkeeping would otherwise corrupt
                # the very snapshot used for the next branch candidate.
                setattr(env, name, value.to(env.device).clone())
        env.is_start = bool(payload["is_start"])
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"])


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: Any) -> None:
        return None
