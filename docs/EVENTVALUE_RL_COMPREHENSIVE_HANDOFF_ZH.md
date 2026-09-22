# EventValue-RL / πRL 项目综合交接报告

> 最后更新：2026-09-23  
> 本地仓库：`/home/jj/RLinf-piRL`  
> 工作分支：`event-smdp-credit`  
> 已推送的最新提交：`9fbdcf0d`  
> 面向对象：接手本项目的新 AI/工程研究者

本文是当前工作的**事实性交接文档**。它包含研究目标、已确定的系统设计、实际代码与实验状态、失败过的路径、HPC 位置、当前瓶颈和下一步执行顺序。除非另有标注，不能将“设计计划”误写成“实验已验证结论”。

---

## 1. 一句话摘要

项目旨在以官方 **π0.5 + πRL（Flow-SDE + PPO）** 为唯一 actor/RL 底座，加入“事件价值 + 受控分支干预”的 credit assignment：Event Observer 产生事件状态，Event-SMDP Value 负责长程价值，same-state action branch 衡量动作改变未来事件价值的程度，再把 credit 在事件内重新分配。

目前，RoboTwin `adjust_bottle` 上的完整**冻结 branch 采集、chunk 时间轴、reference-action transport、PPO 接口、独立 16/8/8 seed 切分**均已跑通；但是独立 test 上的 learned Influence 排序没有优于随机。因此 `lambda=0`，尚未启动非零 Event credit PPO，也没有 EventValue-RL 成功率提升可宣称。当前最高优先级不是调大网络或扩大 PPO，而是验证 bootstrap branch target 是否真的排序真实 SFT continuation 回报。

---

## 2. 研究问题、核心主张与边界

### 2.1 要证明的研究主张

论文应最终证明的不是“事件识别有用”，而是：

\[
\boxed{
\text{事件价值提供长程时间抽象；受控动作干预测量真正改变未来事件价值的动作；}
\text{二者令 Flow-VLA 的 credit assignment 更准确、更省样本。}
}
\]

具体地，actor 仍是 π0.5；πRL 仍负责 Flow-SDE action sampling、simulator interaction 和 PPO。创新模块仅替换/修正 credit estimator，而不重写 VLA 或 RL 框架。

### 2.2 不应再作为主创新的内容

- **CfC 已永久删除**，旧 checkpoint/指标仅是历史对照；
- 旧称“causal tracker / causal event graph”的 GNN 不应被称为因果证据；真正因果证据必须来自同一 simulator state 下不同 action 的受控 branch；
- adaptive horizon / adaptive exploration 最多做附加 ablation，不承担主 claim；
- 不能把 RoboTwin 多阶段单目标任务单独写成“通用 long-horizon VLA 已证明”。最终长程外部验证应补 CALVIN ABC→D 或 LIBERO-Long（优先 CALVIN）。

### 2.3 当前主实验定位

- **RoboTwin `adjust_bottle`**：当前最可运行的机制闭环。对比 πRL-PPO/GAE、Event-SMDP、branch-interventional credit；
- **RoboTwin multi-archetype**：后续用于 task-general Event Observer 预训练/关系泛化，不应向 πRL baseline 引入额外不公平数据；
- **CALVIN ABC→D 或 LIBERO-Long**：后续外部长程验证。CALVIN 必须严格只用 ABC 训练/在线更新，D 只做 zero-update evaluation；
- **ManiSkill3**：适合 clean clone-state intervention proof，但当前主运行资源已转向 RoboTwin。

---

## 3. 目标架构（最终设计）

### 3.1 端到端结构

\[
\text{Instruction}
\rightarrow \text{Role Set}
\rightarrow \text{offline SAM teacher tracks}
\rightarrow \text{online-capable Role-Graph Event Observer }F_E
\rightarrow (z_t^E,b_t,p_t,u_t)
\rightarrow V_E^\pi
\rightarrow \text{matched-state branches}
\rightarrow I_\xi
\rightarrow A_t^{\mathrm{EVI}}
\rightarrow \pi\mathrm{RL\ PPO}.
\]

当前 `adjust_bottle` 实验处于这条链的中段：Observer/RGB student 与 Event Value sidecar 已接入；branch 采集、Influence 离线训练与 policy-relative score 都已接通；**只有 Influence 未满足开闸条件**。

### 3.2 感知与 Observer 的最终协议

当前保留的泛化方向是 **“RGB + 部署可得的实测本体感知”**：

1. 在仿真 replay 中，从 articulation 读取**实测** qpos、夹爪开合、末端位姿及速度。RobotWin 官方 HDF 中的 `joint_action/*`、`endpose/*` 是指令，不可直接当 measured；
2. 不使用 oracle segmentation、深度、物体真值位姿作为 Event Observer 特征；它们只产生监督标签和做评估。这一纪律是为将来真机部署保留的，而不是 UMI 专属约束；
3. 不设置 hand-eye 标定后的常量 tip 原点，也不把 object–tip distance 当主特征。腕相机被看作夹爪局部视角；安装差异以图像归一化、crop/translation augmentation 与 mount token 吸收；
4. object/target/part 的 teacher track 离线由 SAM 3.1 生成：首帧 exemplar 或简单名词短语 grounding、视频 propagation、遮挡/ID 失败后的 re-anchor；SAM 永不进入在线 PPO；
5. 每帧缓存低维视觉关系：角色归一化位置、尺度、速度、相对位置/速度/overlap、object–background residual flow、visibility/reliability/missing；本体感知：归一化 measured aperture、因果导数、plateau、末端线/角速度；多相机分别归一化，再以 mount token 融合；
6. role set 应支持 `manipulated_object / goal_region / articulated_part / handle_or_actuator / tool / gripper`，而不是固定 object-target-gripper；关系原语是 `approaching/contact/attached/co-moving/constrained-motion/aligned/inside-or-on-target/released/articulating/actuated`；
7. 几何可得原语与外观状态变化原语应分 head 报告；`actuated`（按钮按下、开关改变等）可能是最弱 head，应作为 limitation 单列。

### 3.3 Event 输出与监督

Event Observer 输出：

\[
z_t^E,\quad q(e_t\mid x_{\le t}),\quad b_t,\quad p_t,\quad u_t,
\]

其中 (b_t) 是 boundary probability，(p_t) 是 event progress，(u_t) 是不确定性。关系原语可以保留为辅助监督，但最终研究目标是统一 event representation，不是固定 task stage classifier。

当前 RobotWin oracle 可提供 contact/grasp/lift/release/success/boundary 等标签；这些标签不进入部署特征。boundary 必须报 Precision/Recall/F1 或 (F1@\pm k)，不能只报高度不平衡的 accuracy。

### 3.4 Event-SMDP Value

对事件 (E_j=[s_j,\ldots,b_j-1])，真实事件持续时间 (D_j) 下的高层 TD 目标是：

\[
\delta_j^E=R_j+\gamma^{D_j}V_E(z_{b_j}^E)-V_E(z_{s_j}^E).
\]

工程修复后的 **独立 state/chunk Value target** 必须是对每个有效 chunk state 的剩余 reward-to-boundary：

\[
y_t=\sum_{k=t}^{b-1}\gamma^{k-t}r_k+
\gamma^{b-t}m_b\bar V_E(z_b^E),
\]

而不是旧式的把一个事件 advantage 平均加回每个内部 token 的伪 Value target。`m_b` 根据真实 termination 与 truncation 定义决定 bootstrap；真正 task terminal 不 bootstrap，time-limit truncation 的规则必须与 πRL baseline 对齐。target Event Value 使用 EMA target network：

\[
\bar V_E\leftarrow \tau\bar V_E+(1-\tau)V_E.
\]

Observer 可 offline pretrain 后冻结；Event Value 随当前 policy 在线更新（当前 record-only 审计中学习率为零以冻结标签）。

### 3.5 干预与 Influence 定义

在完整 action chunk 的状态 (s_t)，从**同一完整 simulator snapshot**执行 (K\approx4) 个完整 Flow-SDE action chunk branch，短 horizon 终点产生：

\[
Y(s_t,a)=R_{t:t+H}+\gamma^H\bar V_E(z_{t+H}^E).
\]

原始 matched-state action effect：

\[
I(s,a)=Y(s,a)-\mathbb E_{a'\sim\pi}[Y(s,a')].
\]

由于 state-centered 训练只识别同一 state 内候选差异，raw (f(z,a)) 可带任意 (c(z)) 偏移，不可跨时间直接进入 PPO。当前实现使用相同 observation 的未执行 policy reference chunks：

\[
\widehat I(z,a)=f(z,a)-\frac{1}{M}\sum_{m=1}^M f(z,\tilde a_m),
\qquad \tilde a_m\sim\pi(\cdot\mid z).
\]

这些 reference 只增加 Flow-SDE inference，不增加 simulator interaction budget；branch rollout 则必须计入总 interaction budget。

### 3.6 稳定的 credit 方案（尚未启用）

已淘汰的激进 V1：

\[
A_t^{\mathrm{V1}}=D_jw_t\operatorname{sgn}(I_t)|A_j^E|.
\]

问题是仅有几十个 branch label 的 Influence 一旦符号错，就会把正确 SFT action 强制推为负 advantage。

保留的 V2 是“Event 决定总信用，Influence 只做保守重分配”：

\[
c_t=\operatorname{clip}\left(
\frac{I_t-\bar I_j}{\operatorname{Std}(I_{E_j})+\epsilon},-c,c\right),
\]

\[
A_t^{\mathrm{Event}}=\frac{A_j^E}{D_j}+\beta |A_j^E|c_t,
\]

\[
\boxed{A_t=(1-\lambda)A_t^{\mathrm{GAE}}+\lambda A_t^{\mathrm{Event}}.}
\]

原 PPO critic 的 return 必须保持独立 GAE return，不能随 actor 的 mixed advantage 改变。`lambda=0` 必须走 literal GAE path，避免 `0\times\mathrm{NaN}`；当前全体在线测试均为 `lambda=0`。

---

## 4. πRL / π0.5 / PPO 的关系

- **π0.5**：视觉-语言-动作模型，提供 SFT actor，含 Flow-SDE 或 Flow-Noise action generation；
- **SFT checkpoint**：特定 benchmark/task 分布的监督初始化，不是“πRL 方法本身”；
- **πRL**：在 π0/π0.5 基础上用 online RL fine-tuning 的方法/实现，当前使用 Flow-SDE + PPO；
- **PPO**：策略更新算法；官方 πRL baseline 的 flat credit 是 action-level GAE；
- **本方法**：不替换 π0.5 actor、不替换 Flow-SDE、不替换 PPO ratio/logprob；只在 credit/value sidecar 上增添 Event-SMDP 与 intervention correction。

当前可用 RoboTwin SFT：

```text
/home/chefmate/Data/pirl_a6000_run/RLinf-Pi05-RoboTwin-SFT-adjust_bottle
```

它是 `adjust_bottle` 的单任务 SFT，不能被描述为所有 RoboTwin task 的通用 SFT。πRL 是方法，不是一个可随处复用的单一 checkpoint。

---

## 5. 当前实际代码状态

### 5.1 本地与远端位置

| 用途 | 路径 |
|---|---|
| 本地 repo | `/home/jj/RLinf-piRL` |
| A6000 测试 repo | `/home/chefmate/Data/pirl_a6000_run/RLinf-piRL-event-effect-test` |
| A6000 venv | `/home/chefmate/Data/pirl_a6000_run/env_pirl_pi05` |
| RoboTwin 支持代码 | `/home/chefmate/Data/pirl_a6000_run/RoboTwin-RLinf_support` |
| RoboTwin assets | `/home/chefmate/Data/pirl_a6000_run/RoboTwin_v14_dual_gpu_20260831` |
| Curobo source | `/home/chefmate/Data/pirl_a6000_run/runtime/curobo/src` |
| A6000 outputs root | `/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2` |

远端运行时必须设置：

```bash
export REPO_PATH=/home/chefmate/Data/pirl_a6000_run/RLinf-piRL-event-effect-test
export PYTHONPATH="$REPO_PATH:/home/chefmate/Data/pirl_a6000_run/RoboTwin-RLinf_support:/home/chefmate/Data/pirl_a6000_run/runtime/curobo/src"
export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
```

### 5.2 重要实现文件

| 文件 | 作用 |
|---|---|
| `rlinf/algorithms/event_intervention.py` | Event Influence model、policy-relative score 等核心函数 |
| `rlinf/algorithms/event_value.py` | Online Event Value sidecar、Value target/EMA 相关实现 |
| `rlinf/workers/actor/embodied_fsdp_actor_worker.py` | sidecar 前向、branch 监督、reference 对齐、diagnostic `.npz` 写入、PPO 前移除 chunk-clock tensor |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | Flow-SDE branch candidate 与未执行 policy reference action 采样/transport |
| `rlinf/data/schema/embodied_types.py` | `PolicyOutput`、`ChunkStepResult`、trajectory transport 字段 |
| `rlinf/data/schema/embodied_trajectory_builder.py` | chunk-level branch/reference 字段的 trajectory 堆叠 |
| `examples/embodiment/run_robotwin_event_branch_a6000_smoke.sh` | A6000 record-only collection 启动脚本 |
| `examples/embodiment/event_observer/make_robotwin_branch_seed_split.py` | 生成预注册 16/8/8 seed split |
| `examples/embodiment/event_observer/train_influence_from_branch_diagnostics.py` | 训练/验证/审计 Influence |
| `examples/embodiment/event_observer/audit_policy_reference_stability.py` | 2/4/8 reference sample 稳定性审计 |
| `examples/embodiment/event_observer/validate_bootstrap_with_sft_continuation.py` | live RoboTwin branch endpoint 后续 SFT continuation 验证器 |
| `docs/EVENTVALUE_RL_A6000_EFFECT_AUDIT_ZH.md` | 当前 A6000 细节审计报告 |

### 5.3 已实现的关键接口修复

1. **chunk clock 统一**：Event Value、branch 与 credit 以完整 action chunk 为决策时间单位；当前 chunk=5 control steps；
2. **真实 branch 执行完整 chunk**：记录 requested/actual duration、termination、truncation、mask、reward 与 bootstrap 分量；
3. **GAE 隔离**：actor mixed advantage 与 PPO critic target 分开；`lambda=0` 等价 GAE；
4. **sidecar chunk tensor 隔离**：Value/Influence 后在 generic PPO token shuffle 前 pop 掉 `branch_*`、`influence_reference_actions` 等 chunk-clock 字段，修复过 `IndexError`；
5. **proprio/mount 输入合同**：`proprio_time_delta=5`、online mount token 已接入，并做 strict input contract；cache 特征的速度单位应继续审计为每 control step 或明确秒制，不能只信 metadata；
6. **policy reference transport**：rollout worker 从同 observation 采样 (M\) 条非执行 Flow-SDE chunks，actor 用 reference mean 去除 state-only offset；
7. **bootstrap-tail reference 对齐**：branch sidecar 有时有 terminal bootstrap tail（40 chunks），reference 只包含实际 action chunks（39）。代码仅允许这种一个尾行差异，补 placeholder 且写 `policy_reference_available=false`；离线 audit 过滤该行，绝不把零值当 reference；
8. **多卡协议**：auxiliary loss 用全局 valid count 归一化；all-reduce 空样本 rank 贡献零梯度；全局无样本共同 skip update。CPU/Gloo 单元测试已通过，生产多 GPU/NCCL 尚未再次正式验证；
9. **状态可审计性**：branch diagnostic 保存 action、state ID、reward/bootstrap 分量、actual duration、done、valid mask、Event representation、8 reference actions；snapshot/order validator 记录 root/endpoint fingerprint 与 candidate execution position。

### 5.4 最近关键提交

```text
4ba0f522  Prepare independent branch validation split and audits
d8e001e6  Align delayed policy reference chunks for diagnostics
54942779  Document independent branch validation transport fix
14412e22  Disable checkpoints for record-only branch collection
046bb213  Fully disable checkpoints in branch collection mode
9fbdcf0d  Record independent branch ranking audit results
```

此前还有 chunk/cache/Value/GAE 修复提交，详见 `EVENTVALUE_RL_A6000_EFFECT_AUDIT_ZH.md` 第 10 节。

### 5.5 本地验证

以下测试最近通过：

```bash
cd /home/jj/RLinf-piRL
python3 -m py_compile \
  rlinf/workers/actor/embodied_fsdp_actor_worker.py \
  examples/embodiment/event_observer/audit_policy_reference_stability.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit_tests/test_event_intervention.py -q
# 4 passed
bash -n examples/embodiment/run_robotwin_event_branch_a6000_smoke.sh
```

历史 A6000 probe 也验证过：一个完整 200-control-step rollout、chunk=5、reference=8、`lambda=0`、actor LR=0、Event Value LR=0 可正常完成并写出 artifact。

---

## 6. 已跑实验的历史与当前结果

### 6.1 不应引用为正式结果的早期实验

1. **ManiSkill πRL pilot**：官方 Flow-SDE/PPO/GAE 路径可运行，但小预算/配置下 SFT 62.5%，step 25 37.5%，step 50 50%，step 75 43.75%，step 100 28.125%。该 baseline 未稳定复现官方 benchmark，不能用来比较本方法；
2. **早期 Event V1**：step 25/92 success 均为 0。根因高度怀疑是未校准 Influence 直接用 sign 决定 PPO advantage 符号、仅 20/40/60/74 个真实 branch supervision 就全事件外推、Value/Influence 循环污染、action chunk 与 token credit 粒度不对齐。该结果不是 Event-SMDP 思路已被否定的证据；
3. **早期四个 episode LOEO**：有过 train pairwise .970 与 held-out `.774/.600/.481/.768` 的结果。但它们来自开发数据，且重复候选处理、损失、tie 定义均与当前独立审计不同；不能作为最终泛化证据；
4. **旧 branch order 比较**：跨进程比较 root snapshot 不同，不能解释 execution-position bias；不得引用。

### 6.2 当前 pre-registered independent collection

固定 split manifest：

```text
/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/
independent_seed_split_v1/manifest.json
```

seed：

| split | reset seeds |
|---|---|
| train | 335, 371, 1226, 1009, 1051, 885, 1204, 972, 942, 356, 347, 1017, 1221, 75, 1089, 1105 |
| validation | 1033, 820, 260, 167, 52, 448, 1031, 638 |
| test | 223, 1189, 686, 559, 1125, 1215, 1200, 786 |

实际 collection：

- train：`train_v2/raw/` 下 4 个 `.npz`，覆盖 16 个 seed；长驻 Ray 在收尾前因 host RAM 95% 保护退出，但 4 个 artifact 均已存在且 seed 覆盖完整；
- validation：`validation_isolated_v2/`，每 seed 独立 Ray 生命周期，8/8 正常；
- test：`test_isolated_v4/`，每 seed 独立 Ray 生命周期，8/8 正常；
- 每个保存 state 运行 4 candidates，候选 0 有一个重复 action，因此 duplicate merge 后为 3 个独立候选；每个 state 保存 8 policy references；
- branch collection 期间 actor/Event Value 均冻结，`lambda=0`，所以这些不是 PPO 性能结果。

### 6.3 A6000 资源问题及处理

1. **host RAM**：A6000 主机约 62.6GB RAM。一个多 epoch 长驻 Ray 进程运行到第 4 epoch 前，Ray memory monitor 在约 62.19GB（>95%）杀掉 EnvWorker。后续 val/test 改成每个 seed 独立短进程；
2. **磁盘**：RLinf 的 `check_progress` 在最后 epoch 会保存任何 `save_interval>0` 配置，即使 interval 极大。每个 pi0.5 checkpoint 约 20GB；验证 8 job 一度占约 152GB，导致磁盘满、test 卡在 checkpoint。修复为 record-only 脚本默认 `ROBOTWIN_SAVE_INTERVAL=0`。已删除的仅是本次 frozen collection 的 `checkpoints/`，保留原始 `.npz`、SFT/Observer/RGB student/sidecar。最终 test v4 不生成 checkpoint；
3. **当前资源**：测试完成后 GPU 空闲；数据盘曾恢复约 192GB 可用。新运行前必须再次检查 `df -h /home/chefmate/Data`，不要在 record-only job 写 checkpoint；
4. **Ray tmp**：设置在 `/home/chefmate/Data/pirl_a6000_run/ray_tmp`；过期 session 很小，不能解决大 checkpoint 问题。

### 6.4 独立 Influence 训练/验证/test（当前最重要结果）

训练命令使用 train + validation 合并诊断目录，但由显式 validation seed 分组严格分开：

```text
development states = 84
train states = 59
validation states = 25
objective = state_centered_mse
normalized target loss = true
candidate duplicate merge = 4 -> 3
reliable state rule = spread > repeat-action noise
```

test 只用训练好 sidecar 进行一次 `lr=0` 的 evaluate-only 等价审计：

| 指标 | validation | test | test statewise shuffle |
|---|---:|---:|---:|
| tie-aware pairwise accuracy | 0.5208 | **0.4912** | 0.4982 |
| Kendall-\(\tau_b\) | 0.0333 | **-0.0145** | -0.0029 |
| top-1 regret | \(8.30\times10^{-4}\) | \(7.01\times10^{-4}\) | \(9.61\times10^{-4}\) |
| eligible pairs | 48 | 57 | – |
| state-centered MSE | \(9.11\times10^{-6}\) | **\(4.87\times10^{-6}\)** | zero=\(3.68\times10^{-6}\) |

解读：validation 有极弱正信号，但 test pairwise 低于 shuffle、\(\tau_b\) 接近零略负，且相对 MSE 比零预测差。虽然 test top-1 regret 小于 shuffle，这一项不能抵消其余三项的负面证据。**Influence 没有通过 ranking gate。**

### 6.5 标签信号与 reference 稳定性

独立 test 的可靠性 audit：

| 量 | 数值 |
|---|---:|
| test candidate spread median | \(5.61\times10^{-4}\) |
| repeat-action noise median | \(3.24\times10^{-5}\) |
| spread/noise | 约 17 倍 |
| test spread p90 | \(3.87\times10^{-3}\) |
| repeat noise p90 | \(1.12\times10^{-4}\) |

这说明候选 branch target 并非全零，也不完全是 restore/render 噪声。

policy-relative reference stability（test 32 保存 state、128 次 subsample）：

| reference 数 | 相对 8-reference 的 sign flip | score std mean |
|---:|---:|---:|
| 2 | 9.23% | \(5.47\times10^{-4}\) |
| 4 | 7.25% | \(3.11\times10^{-4}\) |
| 8 | 0% | 接近数值零 |

因此若将来开启 PPO，reference 数不应低于 4，优先 8；但这不证明 Influence 与 branch target 对真实任务价值相关。

### 6.6 已有 Observer 前端指标（历史，非本次 test endpoint 证明）

- teacher/cache Event Observer boundary F1 约 0.703；
- RGB student 与 teacher 的整体边界指标接近，历史 Value MAE teacher/online 约 0.1003/0.1090；
- 这些是全局事件预测指标，不足以证明候选之间 \(10^{-4}\sim10^{-3}\) 级 Event Value 差异的排序可靠；
- 因此不可用“Observer F1 合格”排除前端对 Influence 微差排序的影响。

---

## 7. 已确认的问题、未确认假设与禁止动作

### 7.1 已确认的问题

1. **跨 episode Influence 泛化失败**：这是目前唯一可明确的算法结论；
2. **有限统计量**：test 仅 28 high-signal state / 57 eligible pairs，仍不足以做强负结论，但绝不足以开 PPO；
3. **开发集过拟合风险**：开发阶段多次使用旧四 episode。最终 test 只有这次预注册 test v4；不得拿旧 LOEO 覆盖它；
4. **branch 目标可能被 bootstrap 主导**：短 branch reward sparse，(Y\) 很大部分来自 \(\bar V_E(z_{end})\)；当前尚未证明这个 bootstrap 的排序对应实际剩余 task return；
5. **部署资源问题已修复但需保持**：record-only 禁 checkpoint；多 epoch long-running Ray 会 host RAM 累积。

### 7.2 尚不能下结论的假设

以下都是待测假设，不得在论文或报告中写成根因：

- Event Value 的 bootstrap ranking 与真实 SFT continuation ranking 不一致；
- Flow-SDE candidate 动作太相似、branch horizon 太短，或采样位置不在关键决策点；
- 关键事件状态覆盖不足，缺少 contact/adjust/near-success/failure-recovery 的多样性；
- online RGB student/Observer 的 candidate-relative Value 误差影响排序；
- snapshot/restore 低幅数值噪声在 near-tie pair 上仍主导；
- 当前 MLP Influence 与 state-centered MSE 不是最合适的跨 episode objective。

### 7.3 当前明确禁止的动作

在 continuation 验证前：

- 不得开启 `lambda>0`；
- 不得报告 EventValue-RL success、sample efficiency 或 credit improvement；
- 不得用更多 PPO epochs、更大模型或 actor LR 试图“调出”正结果；
- 不得把 test seed 放回 Influence/Observer/Event Value 训练或 checkpoint selection；
- 不得把 branch 额外 simulator transitions从 interaction budget 中排除；
- 不得把 raw (f(z,a))（未 reference-centering）直接用作跨时间 influence；
- 不得用 episode-level random split 替代 task/object/scene/archetype 泛化声明。

---

## 8. 接手后的最高优先级：真实 SFT continuation 验证

### 8.1 必须回答的问题

从同一个 test root state 采样不同完整 5-step candidate chunks。每个 branch endpoint 用固定 SFT 继续执行到 termination，比较：

\[
Y_{\mathrm{bootstrap}}=R_{\mathrm{branch}}+\gamma\bar V_E(z_{\mathrm{end}})
\]

与

\[
Y_{\mathrm{continuation}}=R_{\mathrm{branch}}+\gamma G_{\mathrm{SFT,continue}}.
\]

必须同时报告：bootstrap-vs-continuation pairwise/Kendall/top-1 regret、continuation repeat mean/std、candidate action ID、execution position、terminal/truncation、endpoint RGB/proprio fingerprints。near-tie 或 continuation std 与候选差异同量级的 pair 必须标为不确定或增加 repeat。

### 8.2 现有验证器的限制

现有脚本：

```text
examples/embodiment/event_observer/validate_bootstrap_with_sft_continuation.py
```

已支持：

- full 5-step branch；
- (K\) candidate；
- fixed SFT continuation；
- duplicate first candidate；
- 单一或多个 execution orders；
- 同进程同 root snapshot 的 branch-only order check；
- root/endpoint/action fingerprints。

**但它当前不支持 `--reset-seed` / 直接读取 saved branch state。** 因而下一位接手者应先小幅扩展它：

1. 增加 `--reset-seed`，从单 seed manifest 或直接将 `env.reset_state_ids` 设置为该 seed，再调用 reset；
2. 增加 `--state-chunks`（已有）并针对 test 预注册 seed 选择时间分层位置，例如 5, 15, 25；
3. 第一轮建议 4 个新/预注册 test seed、每个 3 states、每 state 3 independent candidates（另加重复 candidate 作 noise control）、continuation repeats=2；
4. 分支执行顺序应随机/交替，而非永远正序；
5. 使用完全冻结的 Observer、RGB student、**target Event Value** 与 π0.5 SFT；不得用 test 结果重新训练它们。

不要把当前 `test_isolated_v4` branch `.npz` 误当可直接 continuation 的完整 simulator snapshot：它保存的是 state identity / branch label / representation，不保存可安全 restore 的物理 state。若要针对相同 seed/chunk 复验，需在 live env replay 到对应 chunk，再做新 branch；这也是为什么必须记录 reset seed 和 elapsed steps。

### 8.3 continuation 后的决策树

| continuation 结果 | 下一步 |
|---|---|
| bootstrap 排序与真实 continuation 不一致 | 优先修 (V_E\) calibration / target / branch horizon；不要再训 Influence |
| bootstrap 排序有效、但 Influence train/test 仍差 | 扩大独立关键状态覆盖；比较 state-centered MSE 与可靠 pairwise Huber；检查 action normalization、representation 与前端误差 |
| bootstrap 有效、Influence 有稳定 test 优势 | 才开始 conservative Event-PPO，`lambda:0→0.1→0.25`，同时保留 GAE critic return |
| true Influence 与 shuffle/reverse 无差别 | 不能声称 intervention credit，有必要重新定义候选/目标/状态选择 |

---

## 9. 后续实验设计（在当前 gate 通过之后）

### 9.1 oracle 诊断阶梯

严格顺序：

\[
\text{GAE baseline}
\rightarrow \text{oracle event boundary + Event-SMDP uniform}
\rightarrow \text{oracle event + true branch ranking}
\rightarrow \text{predicted event + true ranking}
\rightarrow \text{oracle event + learned Influence}
\rightarrow \text{predicted event + learned Influence}.
\]

解释：

- 2 不优于 1：Event Value、时间尺度或 Event-SMDP credit 公式有问题；
- 3 优于 2：真实 intervention credit 有机制潜力；
- 4 下降：Observer/online frontend 是瓶颈；
- 5 下降：Influence 是瓶颈；
- 6 下降：模块误差相加或 policy drift；
- 必须加入 **shuffle influence** 与 **reverse influence** 两个 falsification control。

### 9.2 公平 PPO 对比

当且仅当 ranking/continuation gate 通过：

- 同一 π0.5 SFT；
- 同一 RoboTwin task、Flow-SDE、PPO 参数、chunk、gamma、评估器；
- πRL baseline 保持 action-level GAE；
- baseline、Observer+GAE、Event-SMDP uniform、temporal/boundary credit、full branch-interventional credit 使用同样 total simulator interaction；
- 真实 branch rollout 必须计入

\[
N_{\mathrm{total}}=N_{\mathrm{actor}}+N_{\mathrm{branch}};
\]

- 初版至少 3 seed，报告 success-vs-env-step AUC、达到阈值成功率所需 transitions（如 \(N_{80},N_{90}\)）、最终 success、稳定性；
- branch learner inference reference 不执行 simulator，可单列 model inference 成本，但不可伪装成额外交互零成本的 branch。

### 9.3 论文 benchmark 计划

1. RoboTwin `adjust_bottle`：先闭环机制与所有 ablation；
2. RoboTwin 另 1–2 个不同 archetype（例如 handover/tool/press），若 πRL 任务配置与 SFT 可用，用于最小跨任务证据；
3. CALVIN ABC→D：所有 SAM cache、oracle label、Observer pretrain、Event Value initialization、branch/PPO online 必须来自 ABC；D 只最终 evaluation；指标沿用 Avg. Subtasks 与 Len-1…Len-5；
4. LIBERO-Long：独立第二长程 benchmark；
5. RobotWin 16-task/multi-archetype Observer 训练只做附录的 representation/generalization，不可让主 πRL 对比因额外数据受益。

---

## 10. 可复现数据与报告位置

```text
# Event Observer / RGB student
/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/observer_chunk5/best.pt
/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/rgb_student_chunk5/best.pt

# Frozen sidecar used to initialize independent Influence fitting
/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/
policy_relative_transport_v3/robotwin_adjust_bottle_branch_a6000_smoke/
checkpoints/global_step_1/actor/eventvalue_sidecars.pt

# Pre-registered seed split
/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/
independent_seed_split_v1/manifest.json

# Independent raw branch data
/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/
independent_branch_validation_v1/train_v2/raw/
/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/
independent_branch_validation_v1/validation_isolated_v2/
/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/
independent_branch_validation_v1/test_isolated_v4/

# Latest independent Influence artifacts
/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/
independent_branch_validation_v1/influence_independent_v1/development_report.json
/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/
independent_branch_validation_v1/influence_independent_v1/test_report.json
/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/
independent_branch_validation_v1/influence_independent_v1/test_reference_stability.json
```

### 10.1 复跑独立 Influence 审计的命令示意

在 A6000 repo 中：

```bash
export REPO_PATH=/home/chefmate/Data/pirl_a6000_run/RLinf-piRL-event-effect-test
export PYTHONPATH="$REPO_PATH:/home/chefmate/Data/pirl_a6000_run/RoboTwin-RLinf_support:/home/chefmate/Data/pirl_a6000_run/runtime/curobo/src"
cd "$REPO_PATH"

# development_links_v1 是 train + validation 的符号链接集合；validation groups 显式给出
/home/chefmate/Data/pirl_a6000_run/env_pirl_pi05/bin/python \
  examples/embodiment/event_observer/train_influence_from_branch_diagnostics.py \
  --diagnostics .../development_links_v1 \
  --sidecar .../policy_relative_transport_v3/.../eventvalue_sidecars.pt \
  --output-sidecar .../influence_independent_v1/eventvalue_sidecars.pt \
  --report .../influence_independent_v1/development_report.json \
  --validation-groups 1033,820,260,167,52,448,1031,638 \
  --epochs 500 --batch-size 32 --lr 3e-4 \
  --objective state_centered_mse --normalized-target-loss \
  --prediction-tie-eps 0 --action-repeat-eps 0 \
  --min-spread-over-repeat-noise 1.0
```

对 test 做 eval 时应禁学习（当前用 `--epochs 1 --lr 0 --overfit-all-states` 只为复用现有报告脚本）；更干净的后续工程改进是给该工具添加显式 `--evaluate-only --model-sidecar`，避免该语义绕行。

---

## 11. 工作区与安全注意事项

- 本地 git worktree 有用户既有未提交文件，不能删除或纳入本项目提交：

```text
D  docs/EVENT_SMDP_INTERVENTIONAL_CREDIT_ZH.md
?? docs/EVENTVALUE_RL_V2_INTEGRATION_REVIEW_ZH.md
?? docs/ICLR_EVENT...md
```

- 当前新增交接文档及 A6000 audit 文档均已在 `event-smdp-credit` 推送；
- 不要删除 A6000 下的 raw `.npz`、split manifest、Observer/RGB student、SFT 或 sidecar；
- 如果磁盘再次紧张，只可在核验 raw artifact 已保留后删除**本次 record-only collection**目录中的 `log/**/checkpoints/`。不要删除未知 `Data/vla`、`Data/agent-sfy`、`Data/astribot` 等用户数据；
- 运行 record-only collection 必须显式保持：

```bash
ROBOTWIN_EVENT_DIAGNOSTIC_RECORD_ONLY=true
ROBOTWIN_EVENT_DIAGNOSTIC_FREEZE=true
ROBOTWIN_EVENT_VALUE_LR=0.0
ROBOTWIN_SAVE_INTERVAL=0
```

- 在任何 Event-PPO 前再次确认：`max_lambda=0` 已解除的唯一依据应是记录在文档中的 continuation + independent ranking gate，而非主观觉得“分支看起来有差异”。

---

## 12. 当前状态清单（供快速接手）

| 项目 | 状态 | 结论 |
|---|---|---|
| π0.5 RoboTwin adjust_bottle SFT | 可用 | 当前所有收集使用同一 checkpoint |
| πRL/Flow-SDE/PPO transport | 已跑通 | 但本轮冻结，不代表 PPO 效果 |
| chunk/Value/GAE 接口 | 已单测 + A6000 probe | 仍需在未来正式多卡 PPO 再验 NCCL |
| same-state branch | 已跑通 | label 有有限动作敏感性 |
| policy-relative reference | 已跑通 | 8 reference 最稳定 |
| Observer/RGB student | 已接入 | 微差排序影响未排除 |
| 16/8/8 independent split | 已完成 | test 未用于训练/选择 |
| Influence development validation | 弱正信号 | 不足以开 gate |
| Influence independent test | 未通过 | pairwise/τ 不优于随机 |
| true SFT continuation validation | **未完成** | 当前最高优先级 |
| Event-PPO \(\lambda>0\) | **未启动** | 当前禁止启动 |
| 论文效果 claim | 不成立 | 尚无 success/sample-efficiency 增益 |

最重要的接手原则：**保持当前 architecture 与 `lambda=0`，先证伪或证实 bootstrap label 的真实 continuation 排序。**
