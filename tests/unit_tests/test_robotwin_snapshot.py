import threading
from types import SimpleNamespace

import torch

from rlinf.envs.sim.robotwin.robotwin_snapshot import restore_robotwin_env, snapshot_robotwin_env


class _Physics:
    def __init__(self):
        self.value = 3

    def pack(self):
        return str(self.value).encode()

    def unpack(self, state):
        self.value = int(state.decode())


class _Scene:
    def __init__(self, physics):
        self.physics = physics

    def get_physx_system(self):
        return self.physics


def test_robotwin_snapshot_restores_physics_and_bookkeeping():
    physics = _Physics()
    task = SimpleNamespace(
        scene=_Scene(physics), run_steps=4, reward_step=2, stage_success_tag=False,
        instruction="lift bottle", info={"score": 1},
    )
    refreshed = []
    task._update_render = lambda: refreshed.append(True)
    subenv = SimpleNamespace(task=task, instruction="lift bottle", lock=threading.Lock())
    env = SimpleNamespace(
        venv=SimpleNamespace(envs=[subenv], global_lock=threading.Lock()),
        prev_step_reward=torch.tensor([1.0]), _elapsed_steps=torch.tensor([4]),
        success_once=torch.tensor([False]), fail_once=torch.tensor([False]), returns=torch.tensor([1.0]),
        is_start=False, device=torch.device("cpu"),
    )
    state = snapshot_robotwin_env(env)
    physics.value = 99; task.run_steps = 40; task.info["score"] = 99
    env.prev_step_reward[:] = 99; env.is_start = True
    restore_robotwin_env(env, state)
    assert physics.value == 3
    assert task.run_steps == 4 and task.info == {"score": 1}
    torch.testing.assert_close(env.prev_step_reward, torch.tensor([1.0]))
    assert not env.is_start and refreshed
