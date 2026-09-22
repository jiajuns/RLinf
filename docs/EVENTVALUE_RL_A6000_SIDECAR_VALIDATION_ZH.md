# EventValue-RL sidecar：A6000 验证审计

日期：2026-09-22  
代码分支：`event-smdp-credit`  
运行平台：A6000（`chefmate`，单张 RTX 6000 Ada，RoboTwin `adjust_bottle`）

## 结论

本次验证通过了 EventValue-RL V2 在**信用估计安全性**与 **RoboTwin 同状态分支恢复**上的四项准入检查。特别地，实际 RoboTwin 稀疏奖励路径的 `take_action_cnt` 曾未被快照；这会让第一个 branch candidate 消耗动作计数，污染下一个候选及主轨迹的截断结果。现已将其纳入任务侧快照，并在 A6000 用实际 50-step action chunk 验证恢复正确。

这不是正式 PPO 训练或正增益结论：`event_mix_lambda` 仍应保持 `0`，直到独立 held-out branch 排序审计合格。

## 已验证的接口约定

- 一个 π0.5 policy action chunk 是一个 Event-SMDP / PPO / branch 的共同时间单位。
- RoboTwin branch 执行完整 chunk，保存 requested/actual duration、termination、truncation、候选 action、状态身份与有效掩码。
- `V_E` 采用从当前 chunk 到事件边界的剩余回报 target；不再把 event advantage 的均分 target 用作 value regression。
- actor 可混合 Event advantage，但原 PPO critic 的 return 始终保持 GAE target。
- `lambda=0` 直接走 GAE 路径，完全不读取 Event tensor，避免 `0 × NaN` 等污染。
- branch target 的可复算字段会随 diagnostics 保存：`branch_rewards`、`branch_bootstrap_values`、`branch_discounted_bootstrap`、bootstrap mask/discount unit、终止/截断、duration、candidate action、state id 与有效掩码。其公式为 `Y = branch_rewards + branch_discounted_bootstrap`。

## A6000 执行结果

### 1. Value target 与 GAE 回归

命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/unit_tests/test_event_intervention.py \
  tests/unit_tests/test_event_credit.py \
  tests/unit_tests/test_event_value_sidecar.py
```

结果：`18 passed`。

其中明确覆盖：

- 两步 reward `[0, 1]`、真终止、`gamma=1` 时两个状态的 `V_E` target 均为 `1`。
- padding 的 event id 为 `-1` 时 target mask 为 false，不参加 value loss。
- 真 terminal 永不 bootstrap；truncation 是否 bootstrap 只遵从 `bootstrap_on_truncation` 显式开关。
- 同一 rollout、`lambda=0` 的 actor advantage 与 critic return 与原 GAE 完全相等。
- `lambda>0` 仅改变 actor advantage；原 PPO critic return 仍保持 GAE return。
- `lambda=0` 允许 Event tensor 为 `None`，验证无效 Event 数值不会污染零混合路径。
- controlled branch return 的回归公式及恢复调用路径。

### 2. 双 rank 辅助网络同步

命令：

```bash
python examples/embodiment/event_observer/validate_event_sidecar_distributed.py
```

结果：`two-rank Event sidecar regression passed`。

该主机只有一张 GPU，因此此项以 CPU/Gloo 启动两个 rank，直接调用生产的 distributed helper；它验证 collective 协议，不是两张物理 GPU 的 NCCL 性能测试。

覆盖：

- 仅 rank 0 有一个有效 branch label、rank 1 只有 graph-connected zero loss：无挂起，两个 rank 的 sidecar 参数、EMA target 与计数相同。
- 全局有效样本数为 0：两个 rank 共同跳过 optimizer；参数不更新、不会出现 rank 间 step 分化。
- 辅助 loss 按全局有效样本数量归一化，而非错误平均各 rank 的局部 mean。

### 3. 真实 RoboTwin 完整 chunk branch

使用 `adjust_bottle`、minimal shader、无 denoiser，候选为两个完全相同的全零 50-step action chunk。实际输出：

```text
elapsed_after_branch        = [0]
elapsed_after_first         = [50]
elapsed_after_restore       = [0]
first_truncated             = [False]
second_truncated            = [False]
take_action_cnt_after_branch= [0]
take_action_cnt_after_first = [50]
take_action_cnt_after_restore = [0]

branch_candidates           = 2
requested_duration          = [[50, 50]]
actual_duration             = [[50, 50]]
repeat_branch_reward_abs_diff = [0.0]
repeat_branch_proprio_abs_max = 0.0
reward                      = [0.0]
rgb_mismatch_fraction       = {main_images: 0.0, wrist_images: 0.0}
```

额外比较了 branch 前后的 SAPIEN physics、任务 bookkeeping、RNG、环境 metrics 的**按值**快照；无差异。因此 branch 恢复后没有改变主 rollout。按值而非整个 pickle 字节串比较是必要的：Torch tensor pickle 会包含 storage 表示，字节不同并不等于状态不同。

### 4. 已修复的真实缺陷

`gen_sparse_reward_data()` 实际使用 `task.take_action_cnt` 管理 horizon，而旧快照只保存 `run_steps`。这会使 branch 的 candidate 0/1 以及之后的主 chunk 看到不同的截断状态。修复为任务快照同时保存和恢复：

```text
run_steps, take_action_cnt, reward_step, eval_success,
stage_success_tag, plan_success, instruction, info
```

同时保留前两项防 alias 修复：CPU bookkeeping tensor 在 snapshot 时 clone；restore 时再 clone，防止 branch 对 snapshot payload 本身的 in-place 污染。

## 非阻塞运行信息

RoboTwin 初始化时输出 `curobo` 与 `pytorch3d` 缺失提示，但本次环境构造、SAPIEN 状态 clone、两个完整 chunk、RGB/实测 proprio 读取和断言均成功；这些提示未导致本验证降级或失败。正式训练前仍建议在训练镜像中补齐或明确禁用相应 optional planner/visualization path。

## 后续准入条件

1. 用固定 SFT actor 收集独立的 matched-state branch 数据，并让 branch transitions 计入总 interaction budget。
2. 先校准 target `V_E`、再训练 `I_xi`；冻结 Observer、RGB student、target value 与 Influence 后，运行 `analyze_branch_diagnostics.py`。
3. 必须报告 return spread、zero predictor MSE、tie-aware pairwise accuracy、Kendall tau-b、top-1 regret、预测方差、成功/失败分层以及同动作重复性。
4. 仅在 held-out ranking 达标后，将 actor 的 `lambda` 从 `0` 保守地增大；原 PPO critic 始终回归 GAE return。

