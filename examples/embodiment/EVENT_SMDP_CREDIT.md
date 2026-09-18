# Event-SMDP interventional credit for πRL

This extension keeps RLinf's π0.5 policy, Flow-SDE sampler, rollout workers,
PPO ratio, and PPO loss unchanged. It adds the `event_smdp_interventional`
advantage estimator only.

Use `maniskill_ppo_openpi_pi05_flow_sde.yaml` as the explicit Flow-SDE π0.5
πRL baseline; it retains stock GAE and actor-critic PPO. The proposed method
is the separate `maniskill_ppo_openpi_pi05_event_smdp.yaml` derived config. It
is not a claim that the stock ManiSkill π0.5 PPO configuration used Flow-SDE.

Before actor advantage calculation, the rollout batch needs sidecar tensors:

| field | flattened semantic shape | producer |
| --- | --- | --- |
| `event_ids` | `[T, B]` | Role-Graph Event Observer boundary decoder |
| `event_values` | `[T+1, B]` | Event Value Critic on observer event state |
| `intervention_influence` | `[T, B]` | same-simulator-state Flow-SDE branches |

The stored tensors use RLinf's normal chunk layout (`[chunks, B, action_chunk]`; `event_values` includes the bootstrap chunk) and are flattened internally. Per contiguous event, the estimator uses `R_j + gamma**D_j * V_E(e_{j+1}) - V_E(e_j)`, then allocates the event advantage across timesteps with a softmax of same-state influence.

The derived configuration uses `loss_type: event_actor`: this is the ordinary RLinf PPO actor loss, but it intentionally does **not** apply Event Value targets to π0.5's original VLA value head. Train the Event Value Critic in the observer sidecar against its SMDP targets and report its explained variance separately.

`rlinf.algorithms.event_intervention` refuses a normal reset as a snapshot substitute. Start in ManiSkill only after its adapter implements lossless `get_state` / `set_state`; do not enable it for RobotWin until its adapter satisfies the same contract.
