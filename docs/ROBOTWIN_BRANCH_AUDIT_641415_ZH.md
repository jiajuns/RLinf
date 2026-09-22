# RoboTwin matched-state branch 审计报告（作业 641415）

## 结论摘要

本次审计的**工程链路已完整跑通**：冻结的 π0.5 SFT actor 在 RoboTwin
`adjust_bottle` 中完成在线 rollout；同一 simulator state 下的 4 个 Flow-SDE
action chunk 分支被执行；每个分支的短期 reward 与目标 Event Value 被记录；原始
`[state, candidate]` return/score 矩阵被写出并离线分析。HPC 作业 `641415` 最终
以退出码 `0:0` 完成。

但审计的**科学结论是否定的**：当前已恢复的 Influence Model (I_\xi) 还不能用于
PPO credit assignment。虽然候选 branch 的真实结果确有可区分差异，模型在 128 个
held-out record-only states 上却没有有效排序能力：tie-aware pairwise accuracy 为
47.64%，Kendall \(\tau_b\) 为 0.0216，且 candidate-0 MSE 是零预测器的 1.87 倍。
因此当前应保持 `event_credit.max_lambda=0`，绝不可把预测 influence 接入 actor
更新或据此声称 interventional credit 已起作用。

这次结果排除了“候选动作完全相同/branch return 全部近零”这个最直接的失败假设，
却确认瓶颈在 Influence Model 的训练与覆盖，而非 simulator clone、Flow-SDE 分支或
诊断数据传输。

## 审计范围与非目标

本作业是一个 **matched-state intervention label identifiability audit**，不是正式
πRL 或 EventValue-RL 成功率实验：

- 目的：检查真实 branch outcome 是否有信号，以及已有 (I_\xi) 能否预测候选
  action chunk 的相对好坏；
- 不更新 π0.5 actor 或原 action-level critic：`actor.optim.lr=0`、
  `actor.optim.value_lr=0`；
- 不让 Event credit 影响 PPO：`event_credit.max_lambda=0`、
  `min_supervision_for_actor=999999`；
- 使用 `event_diagnostics.record_only=true`，不以本审计 batch 更新 Influence Model；
- Observer 为冻结的 RGB/proprio Role-Graph Event Observer；SAM 和 oracle mask 不在
  在线 PPO/branch loop 中运行；
- Event Value 侧车仍按当前 rollout 计算其 SMDP loss，故本报告不应被解释成固定
  \(V_E\) 的独立离线 benchmark。

换言之，本报告只回答：**当前 branch target 是否存在、当前 (I_\xi) 是否已经能
利用它排序**。它不回答 Event-SMDP 是否优于 GAE，也不回答完整方法是否提升任务成功率。

## 作业配置与可复现信息

| 项目 | 值 |
|---|---|
| Slurm 作业 | `641415`，状态 `COMPLETED`，退出码 `0:0` |
| 平台/任务 | RoboTwin `adjust_bottle` |
| 初始化 | `RLinf-Pi05-RoboTwin-SFT-adjust_bottle` |
| GPU / CPU | 2 GPU、24 CPU、240 GB RAM 请求 |
| rollout 并行度 | 64 环境，4 个 rollout epochs |
| actor batch | global batch 512，micro batch 32 |
| Flow 设置 | `flow_sde`、chunk-level reward/logprob |
| branch 设计 | 每 state 4 candidates，chunk interval 10，horizon 10 |
| credit 粒度 | chunk；actor mixing lambda 为 0 |
| 诊断模式 | `record_only=true`，状态级 raw NPZ 输出 |
| 原始数据规模 | 2 个 FSDP rank 文件，128 states，4 candidates/state，512 branch outcomes |

审计使用的运行脚本为
`examples/embodiment/slurm_robotwin_pi05_eventvalue_branch_collect_v2.sh`。
本次修复后，多卡环境明确设置：

```bash
PYTORCH_ALLOC_CONF=expandable_segments:False
```

原因是该 HPC 内核不能在 `expandable_segments=True` 时通过 CUDA IPC 传递 tensor；
此前的 2 卡作业会在 rollout/actor 通信初始化阶段报 `pidfd_open` IPC 错误。修复后
4/4 rollout 正常完成，故这一类错误已被消除。

## 数据与产物位置

HPC 根目录为 `/data/user/leviccdong/EKSF`：

| 产物 | 位置 |
|---|---|
| 原始 rank 0/1 branch 数据 | `outputs/robotwin_branch_audit_2gpu_v2/raw/branch_diag_rank*_opt0_seen128.npz` |
| 机器可读报告 | `outputs/robotwin_branch_audit_2gpu_v2/branch_diagnostic_report.json` |
| 本次 Markdown 报告数据源 | `outputs/robotwin_branch_audit_2gpu_v2/branch_diagnostic_report.md` |
| 本次 sidecar checkpoint | `outputs/robotwin_branch_audit_2gpu_v2/robotwin_branch_audit_2gpu_v2/checkpoints/global_step_1/actor/eventvalue_sidecars.pt` |
| π0.5 actor checkpoint | 同目录 `checkpoints/global_step_1/actor/` |
| Slurm stdout/stderr | `outputs/robotwin_event_branches_v2_641415.{out,err}` |

离线分析命令：

```bash
/data/user/leviccdong/EKSF/env_pirl_pi05/bin/python \
  /data/user/leviccdong/EKSF/code/RLinf-piRL/examples/embodiment/event_observer/analyze_branch_diagnostics.py \
  /data/user/leviccdong/EKSF/outputs/robotwin_branch_audit_2gpu_v2/raw \
  --output-json /data/user/leviccdong/EKSF/outputs/robotwin_branch_audit_2gpu_v2/branch_diagnostic_report.json \
  --output-markdown /data/user/leviccdong/EKSF/outputs/robotwin_branch_audit_2gpu_v2/branch_diagnostic_report.md
```

## 统计结果

### 1. Branch target 是否存在可辨信号？

存在。每个 state 的 4 个候选 return 并没有坍缩到相同数值：

| 指标 | 数值 |
|---|---:|
| state 数 | 128 |
| candidates/state | 4 |
| branch return std，均值 / 中位数 | 0.02939 / 0.01516 |
| branch return range，均值 / 中位数 | 0.07140 / 0.03710 |
| range 的 P90 / 最大值 | 0.14830 / 1.02850 |
| near-zero spread state 比例 | 0.0% |

因此，不应把当前失败归咎于“Flow-SDE 的 4 个候选 chunk 太相似”或“10-step
horizon 完全没有可见后果”。至少在这个 task/策略分布下，matched-state 分支确实
产生不同 future event return。

### 2. 当前 Influence Model 是否可预测这些差异？

不能。全局 held-out record-only 统计如下：

| 指标 | 数值 | 判读 |
|---|---:|---|
| candidate-0 model MSE | 0.002095 | 不够好 |
| 零预测器 MSE | 0.001122 | 更低 |
| model / zero MSE | **1.867** | 比恒为零更差 |
| tie-aware pairwise accuracy | **47.64%** | 近随机，低于 50% |
| Kendall \(\tau_b\) | **0.0216** | 基本无全排序相关性 |
| top-1 exact best | 28.91% | 略高于随机 25%，但不足以形成结论 |
| top-1 regret，均值 / 中位数 | 0.02841 / 0.00876 | 有实际错选代价 |
| predicted-score std，均值 | 0.00204 | 分数非完全常数，但尺度偏小 |
| predicted near-constant states | 0.78% | 主要问题不是输出全零，而是排序错误 |

这也解释了此前 `influence_loss≈0.003` 为什么不能作为正面证据：MSE 的绝对量很小，
但相对零预测器反而更差。后续所有训练日志必须同时报告 model-vs-zero MSE ratio，不能
再孤立报告 influence loss。

### 3. 成功/失败分层

128 个 state 中仅 1 个 state 有任一候选 branch 在短 horizon 内成功；其余 127 个均
没有成功候选。失败层上的排序表现更能代表当前分布：

| 分层 | states | pairwise | \(\tau_b\) | model/zero MSE |
|---|---:|---:|---:|---:|
| candidate 0 failure | 127 | 47.36% | 0.0150 | 2.820 |
| 任一候选成功 | 1 | 83.33% | 0.913 | 0.736 |

后一行只有一个 state，不能解释为模型在成功状态上有效。实际情况是：当前 collection
分布几乎全由失败/非终止 state 构成，成功事件对 (I_\xi) 的监督极端稀缺。

### 4. Event 覆盖

所有 128 个记录 state 都被当前 Observer 赋为 `event_id=0`。这至少表明在本次短
rollout 窗口与现有 `boundary_threshold=0.5` 下，没有预测到可用 boundary。故本审计
只验证“同一当前 event 内的 action-chunk branch 排序”，**没有**验证跨多个语义 event
的 SMDP 时间抽象，也不能用来报告 event-specific influence 差异。

## 正确解释与决策

本次审计支持以下两条事实：

1. simulator state clone、4-way Flow-SDE branching、future Event Value bootstrap、
   raw diagnostic transport、rank 合并与统计计算均正常；
2. action candidates 导致的 return 差异真实存在，值得进一步学习。

本次审计不支持以下主张：

1. 当前 Influence Model 已经学会 causal ranking；
2. branch intervention 已经改善 PPO credit assignment；
3. Event Observer 当前已在在线 rollout 中形成多事件分割；
4. EventValue-RL 已经优于 πRL/GAE。

当前硬性决策：

```text
不要启动 full interventional PPO；
保持 A_t = A_t^GAE（event mixing lambda = 0）；
不要将这 128 个 state 的 MSE/成功率写成方法增益。
```

## 建议的下一阶段

1. **固定 SFT actor，先训练/验证 Influence。** 以 state 为单位切分 train/validation/test，
   不允许同一 base state 的不同 candidates 跨 split；先收集约 500–1000 base states
   （2000–4000 branch outcomes），训练 (I_\xi)，并保留独立 record-only holdout。
2. **增加决策状态覆盖。** 当前 127/128 state 无成功候选，需按 event progress、接近
   成功/失败恢复、低 Event Value 与高 branch spread 分层采样，而不只是等间隔 chunk。
3. **修复/标定在线 boundary。** 对 SFT 和 early πRL rollout 用 simulator oracle 重新
   测 `Boundary Precision/Recall/F1@±k`；若仍全部为 event 0，先解决 Observer 校准或
   boundary threshold，再谈 Event-SMDP。
4. **采用 oracle 诊断阶梯，而非直接 full V2。**
   `GAE → oracle boundary + uniform Event-SMDP → oracle event + real branch ranking →
   predicted event + real ranking → oracle event + learned Influence → predicted event + learned Influence`。
   每级再加入 shuffle-influence 与 reverse-influence falsification。
5. **建议的准入门槛（研究决策门，不是论文结论）。** 在独立 state holdout 上同时达到：
   model/zero MSE < 1、pairwise accuracy 明显高于 50%、正的稳定 \(\tau_b\)，且在
   success/failure 分层中不发生反转；通过后才从
   \(\lambda=0\rightarrow0.1\rightarrow0.25\) 保守开启 residual Event credit。

## 代码变更记录

本审计期间已推送到用户 fork 的 `event-smdp-credit` 分支：

| 提交 | 内容 |
|---|---|
| `3cfb58c7` | 修复 branch diagnostics 的 event-id `[C,B,K]` 对齐 |
| `dbaf0de3` | 重复 record-only collection 不再覆盖旧 `.npz` |
| `219147bd` | HPC 多 GPU IPC 禁用 `expandable_segments` |

这些修复只提高审计数据的正确性与可重复性；它们不改变 π0.5 actor、官方 PPO 更新规则或
当前实验的 credit estimator，因此不会制造方法性能增益。
