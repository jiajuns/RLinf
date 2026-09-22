# EventValue-RL：统一 action-chunk 时间语义与安全接入说明

更新日期：2026-09-22；适用分支：`event-smdp-credit`。

## 结论

本次修改**不更换 π0.5、PPO、Role-Graph Event Observer、Event Value 或 Influence Model 的网络结构**。它只修复 V2 第一版中最危险的接口不一致：PPO 将一个 π0.5 输出的 action chunk 当作一个 RL 转移，但 Event Value 曾把该 chunk 展开成多个伪时间步，而分支又只执行短前缀。现在统一规定：

\[
\boxed{\text{一个完整 π0.5 action chunk}=\text{一个 PPO 转移}=\text{一个 }V_E\text{ 转移}=\text{一次受控 branch 的执行单位}}
\]

因此正式实验在 Influence 验证通过前，`Event-SMDP residual` 的混合系数固定为 \(\lambda=0\)，PPO 的 actor advantage 严格退化为官方的 chunk-level GAE；Observer、在线 \(V_E\) 和 branch sidecar 可以继续收集/训练，但不能改变策略更新方向。

## 为什么必须统一

π0.5 对同一观测采样的是一段长度 \(K\) 的 Flow-SDE action chunk（RoboTwin `adjust_bottle` 当前 \(K=50\)）。在 `reward_type=chunk_level` 下，RLinf 已将 token reward 求和、将 token done 取最大值，得到一个 chunk-level PPO 转移。旧实现却存在两处偏差：

1. branch 配置可用 `horizon=10`，只执行 action chunk 的短前缀，却用这个结果评估整段候选 chunk；
2. 在线 Event Value 更新将 chunk reward/done/event id 展平为 \(K\) 个 action token，等价于把同一个观察状态人为重复 50 次。

这会同时造成 \(\gamma\) 的幂、SMDP event duration、branch bootstrap 和 PPO credit 的时间含义不同。即使 Influence 排序正确，也无法解释其 credit 对应的是哪一个控制决策。

## 代码实现与数据流

### 1. 完整 chunk branch

实现位置：`rlinf/envs/sim/robotwin/robotwin_env.py::RoboTwinEnv.branch_step`、`rlinf/workers/env/env_worker.py`。

branch 输入固定为：

```text
[batch, candidates, K, action_dim]
```

其中每一个候选从同一个 RoboTwin/SAPIEN 快照恢复，执行完整 \(K\) 步 action chunk；结束后无条件恢复 live simulator state。branch 不再接收 `horizon` 参数；脚本中的旧 `event_branch.horizon=10` 已删除。

每个候选保存：

| 字段 | 含义 |
|---|---|
| `branch_rewards` | 完整 chunk token reward 之和，和 PPO `chunk_level` reward 相同 |
| `branch_horizons` | 实际执行 action-step 数；为兼容旧 transport 名称保留 |
| `branch_requested_steps` | 该 candidate 请求的完整 chunk 长度 \(K\) |
| `branch_terminations` / `branch_truncations` | 终止/截断标志 |
| `branch_valid` | 数值、图像和 measured proprio 均可用的候选掩码；terminal 本身仍是有效样本 |
| `branch_mask` | 此 rollout chunk 是否真的执行了 branch（跳过的 chunk 为 false） |
| `branch_state_ids` | `[reset_seed, elapsed_action_steps]`，只用于身份审计、去重和 split，不进网络 |
| `branch_actions` | Flow-SDE 产生的完整候选 action chunks |
| `branch_main_images`、`branch_wrist_images`、`branch_measured_state16` | endpoint 处的在线可得观测，用于 target \(V_E\) |

终止候选的 target 定义为：

\[
Y_m=R_m+\gamma^{1}(1-\mathbb{1}_{\mathrm{terminal},m})\bar V_E(z_{m,\mathrm{end}}).
\]

也就是说 terminal/truncated branch 的 bootstrap 被严格置零；由于时间单位是 **chunk**，每个 branch 只前进一个 RL/SMDP 单位，故 bootstrap 指数恒为 1，而不是把原始 50 个控制 tick 错当成 50 个 RL 步。`actual_duration_steps` 与 `requested_duration_steps` 仍完整保存，用于审计环境实际执行；`reward`、未折扣 bootstrap、折扣 bootstrap 和 `bootstrap_discount_units` 也会写进 sidecar，离线能逐项复算 \(Y_m\)。当前 RoboTwin vector API 对完整 sequence 只给出 chunk 末端 done，因此 `actual_duration_steps` 记录环境实际报告的 chunk 执行长度；若环境报告 chunk 内的早停，则取第一个 done 的位置加一。

### 2. Event Value 也在 chunk 时钟上更新

实现位置：`rlinf/workers/actor/embodied_fsdp_actor_worker.py::_populate_learned_event_sidecar`。

旧式 action-token 展平已删除。现在训练 \(V_E\) 时：

```python
chunk_reward = rewards.sum(dim=-1)       # [C, B]
chunk_done = dones.max(dim=-1).values    # [C + 1, B]
chunk_event_id = event_ids[..., 0]       # [C, B]
V_E(z_0 ... z_C)                         # z 每个 chunk boundary 一个
```

`event_smdp_credit` 因此沿 `C` 个完整 chunk 计算真实 event duration \(D_j\)（单位为 chunk），而不是沿 `C × K` 个重复 token 计算。PPO 保持既有接口：为兼容通用 batch 结构，sidecar 输出仍可填充为 `[C+1,B,K]`，但在 `chunk_level` 预处理时只取 token 0；其余重复项不会参与 SMDP target。

### 3. `proprio_time_delta` 与 mount token 合同

实现位置：

- `rlinf/algorithms/event_value.py`
- `examples/embodiment/event_observer/train_event_observer.py`
- `examples/embodiment/event_observer/train_adjust_bottle_rgb_student.py`

线上 RGB student 的相邻帧现在是相邻 action chunk boundary。因此 measured proprio 的因果导数必须使用同一间隔：当前 `adjust_bottle` 正式合同设为 `proprio_time_delta=50.0`（即 \(K=50\) 个控制步），不是旧脚本的 `5.0`。RGB student 和 Event Observer checkpoint 都必须保存：

```json
{
  "online_input_contract": {
    "version": 1,
    "proprio_time_delta": 50.0,
    "mount_token": 2
  }
}
```

线上 `infer_images` 会显式构造长度为 `[batch,time]` 的 mount token，并传给 frozen Observer，不能再以 `None` 隐式跳过 mount embedding。加载时会交叉检查：Observer metadata、RGB student metadata 与 runtime config 三者的 delta/token 必须完全一致。正式脚本设置 `strict_input_contract=true`；没有 metadata 的旧 checkpoint 只允许以 `strict_input_contract=false` 做历史诊断，**不得用于正式结果**。

这意味着需要用相同 50-step chunk boundary 采样/重建 offline cache，并以 `--proprio-time-delta 50.0 --online-mount-token 2` 重训并导出 Observer 和 RGB student。仅修改线上配置而不重训 offline frontend 不构成输入一致性。

### 4. 多卡 sidecar 一致性与恢复

实现位置：`EmbodiedFSDPActor`。

新增机制：

1. rank 0 在 init/resume 后广播 `V_E`、EMA target critic 和 Influence Model 参数/缓冲区；这样 Influence MLP 不会在不同 rank 各自随机初始化。
2. 每次 `V_E` 与 Influence 反向传播后，对 auxiliary gradient 做 all-reduce 平均，随后每个 rank 使用同一梯度执行 AdamW step。
3. sidecar checkpoint 升级为 `eventvalue_rl_sidecars_v2`，保存/恢复 `event_value_optimizer`、`influence_optimizer`、EMA target、累计 branch 计数、时间单位、delta 和 mount token；加载 optimizer state 后迁移 moment 到 sidecar GPU。
4. 恢复时拒绝非 `full_action_chunk` 的 sidecar，或 delta/mount 不匹配的 sidecar。

这与主 π0.5 FSDP optimizer 分离；没有修改官方 actor/PPO 参数同步路径。

## Influence target、掩码与审计

对有效候选集合 \(\mathcal M_s\)，candidate 0 表示实际执行的 Flow-SDE chunk：

\[
I_0(s)=Y_0(s)-\frac{1}{|\mathcal M_s|}\sum_{m\in\mathcal M_s}Y_m(s).
\]

只有 `branch_mask=true`、candidate 0 有效且至少有两个有效 candidate 的 state 才参与 `I_\xi` loss。无有效 supervision 的 rollout 不调用空掩码 loss，也不会计入 calibration 数量。分支端点 value 使用 EMA target \(\bar V_E\)，而非正在被快速更新的 online \(V_E\)。

诊断 sidecar (`branch_diag_rank*_opt*_seen*.npz`) 现保存：完整候选动作、`state_ids`、event id、候选有效掩码、request/actual duration、terminal/truncated、reward、bootstrap、discounted bootstrap、最终 return、预测 score、success 标签和 gamma。`analyze_branch_diagnostics.py` 会剔除候选不完整的 state，并额外报告实际时长、终止/截断比例；它继续报告 return spread、zero baseline MSE、tie-aware pairwise accuracy、Kendall-\(\tau_b\)、top-1 regret、prediction collapse 和 success strata。

## Event advantage 的安全开关

使用 `algorithm.adv_type=event_smdp_residual`：

\[
A=(1-\lambda)A^{\mathrm{GAE}}+\lambda A^{\mathrm{Event}}.
\]

`_event_credit_mix_lambda()` 增加硬门：默认 `require_ranking_validation=true` 时，除非配置明确给出 `ranking_validation_passed=true`，否则永远返回 \(\lambda=0\)。即使已经有足够 branch labels、足够 optimizer step，actor 仍是原始 GAE。正式脚本现在显式设置：

```text
event_credit.max_lambda=0.0
event_credit.require_ranking_validation=true
event_credit.ranking_validation_passed=false
```

只有在冻结/held-out sidecar 验证集通过预注册门槛后，才创建新的可复现实验配置并改为：

```text
ranking_validation_passed=true
max_lambda=0.10   # 第一阶段
min_supervision_for_actor>=预注册数量
warmup_optimizer_steps=...
ramp_optimizer_steps=...
```

建议第一轮最多 \(\lambda=0.1\)，再以 `0.25 → 0.5` 逐步消融；不建议直接为 1。每次 branch transition 必须计入总 simulator interaction budget。

## 排序验证与 oracle 诊断阶梯

允许开启 Event advantage 的最低条件不是小 `influence_loss`，而是**从未用于 Influence 训练的 state** 上至少证明：

- 候选 return spread 显著高于 tie epsilon；
- `model MSE < zero predictor MSE`；
- tie-aware pairwise accuracy、Kendall-\(\tau_b\) 和 top-1 regret 明显优于随机/常数；
- 在 success/failure strata 均不完全失效；
- 记录 candidate actions 与 state identity 后可复查没有 state leakage。

随后按以下阶梯定位，不应从 GAE 直接跳到 full V2：

1. GAE baseline；
2. oracle event boundary + Event-SMDP uniform；
3. oracle event + **真实** branch ranking；
4. predicted event + 真实 branch ranking；
5. oracle event + learned Influence；
6. predicted event + learned Influence；
7. shuffle influence 与 reverse influence 两个 falsification control。

解释规则：2 不优于 1 指向 Event Value/时间尺度/credit 公式；3 优于 2 才说明 intervention 本身有潜力；4 降低说明 Observer；5 降低说明 Influence；6 降低说明模块误差或 policy drift。真实 Influence 若不优于 shuffle/reverse，不能宣称 causal credit 生效。

## 当前已知证据与不应作出的结论

历史 2-GPU RoboTwin audit `641415` 只有 128 个 state × 4 candidates；return spread 并未塌缩（mean std 约 0.0294），但当时 Influence 的 held-out 排序无效：candidate-0 model MSE 约 0.00209，高于 zero predictor 的约 0.00112，tie-aware pairwise accuracy 约 0.476，Kendall-\(\tau_b\) 约 0.022。这证明“需要校准”，不证明 Event 方法无效，也绝不支持开启 actor credit。新的完整-chunk audit 必须与旧 short-prefix audit 分开存放、单独报告。

## 运行顺序

1. 以 50-step chunk boundary 重建/采样 benchmark 内的 offline cache；重训 RGB student 与 Observer，导出含 input contract 的 checkpoint。
2. 运行 `event_smdp_residual` 且 \(\lambda=0\)：检查 sidecar 能加载、各 rank 的参数/计数一致，正式 PPO 行为仍等价 GAE。
3. 冻结 actor 或保持 actor lr 为 0，使用 `record_only=true` 收集 full-chunk held-out branch audit；运行 `analyze_branch_diagnostics.py`。
4. 先做 oracle diagnostic ladder 与 shuffle/reverse controls；满足预注册 ranking 门槛才写入 `ranking_validation_passed=true`，从 \(\lambda=0.1\) 开始 paired seed 训练。

## 本地验证

本次已完成：

- 对修改的 Python 模块执行 `python3 -m py_compile`；
- 执行 `git diff --check`；
- 以合成 2-state/4-candidate audit sidecar 运行 `analyze_branch_diagnostics.py`，验证新增有效候选、时长和 terminal 字段能被读取并产生报告。

完整 RoboTwin smoke 仍必须在具有 RoboTwin/SAPIEN/π0.5 runtime 的 HPC debug 分区执行；本地工作站不具备该环境，不能把静态检查误报为 simulator 端到端成功。
