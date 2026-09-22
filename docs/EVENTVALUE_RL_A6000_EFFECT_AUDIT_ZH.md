# EventValue-RL：A6000 机制校准与效果审计报告

更新日期：2026-09-22  
分支：`event-smdp-credit`  
状态：**已证明小样本排序可学习性；跨 episode 稳健性、严格 branch 重复性与 PPO 收益尚未建立，仍不满足开启 Event credit / Event-PPO 的条件。**

## 1. 本轮要回答的问题

本轮不是正式比较 πRL 与 EventValue-RL 的成功率，而是在更新 actor 之前，依次确认：

1. Event Value、PPO-GAE、完整 action chunk 及 simulator snapshot 的时间单位和恢复语义是否一致；
2. Role-Graph Event Observer 的离线训练输入与在线输入合同是否一致；
3. 在同一 simulator state、不同 Flow-SDE action chunk 的受控分支中，候选动作是否真的带来超过重复执行噪声的后果差异；
4. 冻结 SFT actor 后，Influence Model (I_\xi(s,a,z^E)) 能否在未见 episode 上正确排序这些候选动作。

只有第 4 项通过，才允许把 Event credit 以小系数混入 PPO advantage；此前固定：

\[
\lambda=0,\qquad A_t=A_t^{\rm GAE}.
\]

因此，本报告中的训练、采集和 sidecar 更新**均不构成 Event-PPO 效果实验**，也没有改动 π₀.₅ SFT actor。

## 2. 当前系统与数据协议

在线主干保持官方 RLinf 的 π₀.₅ / Flow-SDE / PPO 路径。Event 模块是独立 sidecar：

\[
\text{RGB + measured proprio}
\rightarrow F_E
\rightarrow z_t^E
\rightarrow V_E(z_t^E), I_\xi(z_t^E,a_t).
\]

- Actor：冻结的 RoboTwin `adjust_bottle` π₀.₅ SFT checkpoint；本轮 `actor.optim.lr=0`，`actor.optim.value_lr=0`。
- 环境：RoboTwin `adjust_bottle`。
- 动作和 SMDP 单位：完整 action chunk。
- 诊断 chunk 长度：5 个控制步；每个 cloned state 采样 4 个 Flow-SDE candidate chunk。
- 说明：5-step 是为了让约 200-step episode 有足够多的事件决策点的**机制可辨识性 pilot**。它不是官方 50-action-chunk πRL 的复现结果，也不能与官方 50-chunk 指标直接混合报告；最终公平对照必须让 baseline 和 Ours 使用同一 chunk 设置。
- Event credit：`max_lambda=0`，`ranking_validation_passed=false`，`min_supervision_for_actor=999999`。
- 输入：SAM 3.1 离线产生的 RGB role-track 特征 + simulator replay 读回的 measured proprio。oracle contact/grasp/lift/success 仅作标签和评估；不把深度、oracle segmentation 或物体真值输入 Observer。

## 3. 已完成的接口与恢复修复

### 3.1 时间单位与 Value target

实现已统一为完整 action chunk：

- branch 必须执行完整请求 chunk，并保存实际控制步长度、终止/截断、有效候选掩码；
- PPO/GAE、Event-SMDP Value 与 branch bootstrap 折扣均以一个 chunk 为一个决策转移，折扣指数为 1；
- 原始控制步长度单独保存，不能错误地把 chunk-level \(\gamma\) 写成 \(\gamma^{50}\)；
- Event Value 使用独立的“当前 chunk 至下一个 event boundary/有效末端”的回报 target；
- termination 永不 bootstrap；truncation 是否 bootstrap 由显式配置控制，本轮为 `false`；
- 原 PPO critic 始终使用独立的 GAE return，Event residual 即便未来开启也不能污染其 target；
- `lambda=0` 直接走 GAE 分支，避免 `0 × NaN` 污染。

### 3.2 snapshot 缺陷与修复

旧 branch 审计发现 RoboTwin 的 `take_action_cnt` 未纳入 snapshot。该计数参与 horizon/截断判断，因此旧候选分支和主轨迹可能不在同一截断状态下。

修复后：

- snapshot 保存并恢复 `take_action_cnt`；
- CPU tensor 深拷贝，避免 snapshot aliasing；
- 实际 RoboTwin 50-step 零动作恢复检查确认：分支后计数为 0、主轨迹后为 50、恢复后回到 0；重复执行 reward/proprio/RGB 一致。

因此，**快照修复之前的所有 branch 排序数字只应当保留为受已知缺陷影响的历史记录，不能用于方法结论。**

### 3.3 50-step cache 合同修复

cache builder 修复了以下问题：

- 下采样后 `mount_tokens`、`valid_mask` 与特征时间轴未同步；
- 非整除轨迹遗漏 terminal frame；
- producer 校验错误使用了 camera 数量而非 track 时间维；
- 过去可能只在后续 RGB student 训练时才暴露 HDF RGB 截断；现在 cache 构建阶段逐一打开全部下采样后的 `head_camera/right_camera` 图像帧，缺失即拒绝写 cache；
- 169 个配对 HDF 完整性审计发现一条损坏文件 `episode_seed_10485.hdf5`，已从 HPC 精确重传并复验。

新可信 cache：

```text
/home/chefmate/Data/pirl_a6000_run/data/
robotwin_adjust_bottle_event_cache_chunk5_v2_effect_verified2
```

共 217 条 episode cache。旧 `effect_pilot` / `effect_verified` cache 与其模型产物不再作为本轮结论证据。

## 4. 前端训练结果

训练从上述 verified cache 进行，并将 `control_step_stride=5`、`proprio_time_delta=5.0`、`online_mount_token=2` 写入 checkpoint 合同。

### 4.1 Event Observer

checkpoint：

```text
/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/
observer_chunk5/best.pt
```

best checkpoint（epoch 16）的 episode-level validation 指标：

| 指标 | 数值 |
|---|---:|
| boundary precision | 0.6905 |
| boundary recall | 0.7160 |
| boundary F1 | 0.7030 |
| boundary F1@±3 chunks | 0.9576 |
| progress MAE | 0.1710 |
| cached return Event Value MAE | 0.1003 |

注意：这只是单任务 `adjust_bottle` 的 deterministic episode split，用于输入/优化诊断，**不是跨 task、跨 object 或 sim-to-real 泛化结果**。训练日志中的负 `train_loss` 来自可学习不确定度的 Gaussian NLL 项可以为负；判断质量应优先看 F1、MAE 与独立在线审计，不应据此宣称模型异常或成功。

### 4.2 RGB student

checkpoint：

```text
/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/
rgb_student_chunk5/best.pt
```

3 个 epoch 后：

| epoch | validation teacher-feature MAE |
|---:|---:|
| 0 | 0.03997 |
| 1 | 0.02465 |
| 2 | 0.02061 |

此处 teacher 是 SAM cache 导出的低维视觉特征，而非 oracle mask。

## 5. A6000 运行环境修复

首次真实 RoboTwin calibration 在环境初始化失败，根因是 A6000 venv 缺少 Curobo；后续 Ray 错误只是 worker 死亡的连带现象。

处理：

1. 从 HPC 的已运行 πRL 环境确认 Curobo 版本为 `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`；
2. A6000 使用同一版本源码；
3. A6000 缺少 `ninja`，已补齐；
4. A6000 无 CUDA toolkit 可 JIT 编译，因此从相同 PyTorch/CUDA ABI 的 HPC 环境传入已构建的 Curobo `.so` 扩展；
5. 运行脚本将 Curobo `src` 和 `~/.local/bin` 注入 Ray worker 的 `PYTHONPATH/PATH`。

真实 import 已通过：

```text
curobo_import_ok Pose MotionGen .../kinematics_fused_cu...so
```

初次双环境校准的 rollout 和 sidecar 保存成功，但 Ray 因宿主内存阈值杀死 worker，故不用于独立效果结论。后续正式机制 pilot 改为 1 environment，完整退出。

## 6. 回归与分布式检查

在 A6000 Python 环境运行：

```bash
pytest -q \
  tests/unit_tests/test_event_intervention.py \
  tests/unit_tests/test_event_value_sidecar.py \
  tests/unit_tests/test_event_credit.py
```

结果：**18 passed**。

此前已覆盖的检查包括：

- 两步 reward、terminal、padding 的 Event Value target 回归；
- GAE isolation：`lambda=0` 与 GAE advantage/returns 一致，且 Event 值无效时不污染零混合路径；
- 分布式空样本、全局空样本与全局归一化协议；
- 完整 chunk 的 snapshot/restore、duration、termination/truncation 与 return 分解字段。

双 rank 协议验证使用 CPU/Gloo；它证明 collective/空样本逻辑，不等价于完整双 GPU NCCL 生产吞吐测试。

## 7. 第一轮 branch 数据与信号审计

### 7.1 旧 candidate-0-only 校准（历史诊断）

修复 snapshot 和输入合同后，曾以双环境收集：320 cloned states、1280 candidate outcomes。该运行在训练后被 Ray host-memory 阈值中断，但诊断与 sidecar 已落盘。

主要发现：

- 平均 candidate return range：0.002089；
- 同动作重复 absolute return difference 均值：0.000208；
- 说明候选后果差异大于典型重复噪声，分支标签并非全是零；
- 但 online Influence 仅用 candidate 0 回归，却为四个 candidate 排序，得到 \(\tau_b=-0.0189\)、pairwise accuracy 0.337。

这暴露工程问题：四个真实 candidate outcome 中只有一个用于监督，导致昂贵的 branch 监督被浪费。该版本不能用于效果结论。

### 7.2 修复后 all-candidate 单环境校准

实现改为：对同一 state 的所有有效 action chunk 都使用中心化 matched-state target：

\[
I(s,a_k)=Y(s,a_k)-\frac{1}{K}\sum_{k'=1}^{K}Y(s,a_{k'}).
\]

并将冻结的 online Event representation 写入 branch artifact；它不包含 RGB 原图或 simulator oracle state，允许后续离线 Influence 的 train/validation。

单环境完整退出的结果：

| 项目 | 值 |
|---|---:|
| cloned states | 160 |
| candidate outcomes | 640 |
| candidates/state | 4 |
| chunk length | 5 control steps |
| actor learning rate | 0 |
| Event mix lambda | 0 |
| Event Value SMDP loss（当次） | 0.0540 |
| Influence loss（当次） | 0.0059 |
| success_once（4 rollout trajectories） | 0.5 |

产物：

```text
/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/
calibration_all_candidates_raw/
calibration_all_candidates/robotwin_adjust_bottle_branch_a6000_smoke/
  checkpoints/global_step_1/actor/eventvalue_sidecars.pt
```

`success_once=0.5` 是仅四条 rollout 的 SFT 诊断值，不是最终成功率估计。

## 8. 分组 holdout Influence 训练与结果

新增离线训练工具：

```text
examples/embodiment/event_observer/
train_influence_from_branch_diagnostics.py
```

它读取 `state_representations`、candidate action chunks、branch returns 与 state identity，以 reset seed 作为 episode 级 group 做 split，避免随机拆散同一 episode 的高度相关 decision states。

本轮 split：

| split | reset seeds | states |
|---|---|---:|
| train | 57, 852, 978 | 120 |
| validation | 49 | 40 |

训练 100 epoch 后最佳 validation MSE：

\[
8.04\times10^{-6}.
\]

但真正决定能否用于 credit 的排序指标为：

| 指标 | 值 |
|---|---:|
| tie-aware pairwise accuracy | **0.3019** |
| eligible non-tie pairs | 106 |
| top-1 regret mean | 0.000410 |

**当前验证未通过。** 该模型尚未展示可用的跨 episode 候选排序；MSE 的下降可能主要反映中心化 return 的小尺度，而不是模型学到了候选动作优劣。此处的 0.3019 尚未分解 predicted ties/concordant/discordant，也尚未与状态内随机对应基准比较，因而不能据此声称模型“反向排序”，更不能说明 branch 标签本身无用。当前 Influence Model 不应接管、也不应重新分配 PPO advantage。

### 8.1 后续补充：小样本拟合诊断（v2）

报告初版后的诊断工具补齐了中心化 target 的 zero-predictor MSE、预测平局、concordant/discordant、Kendall-tau-b 与 statewise candidate-correspondence shuffle 基准，并在同一组 episode-level split 上重新训练。结果如下：

| 指标 | train（120 states） | validation（40 states） |
|---|---:|---:|
| zero-predictor MSE | `1.73e-6` | `1.45e-6` |
| model MSE | `2.03e-6` | `8.04e-6` |
| model / zero MSE | `1.17` | `5.54` |
| eligible non-tie pairs | 348 | 106 |
| concordant / discordant | 69 / 94 | 11 / 20 |
| predicted ties | 185 | 75 |
| tie-aware pairwise accuracy | 0.198 | 0.104 |
| Kendall-tau-b | -0.058 | -0.068 |
| top-1 regret mean | `7.48e-4` | `4.10e-4` |

在 validation 上对每个 state 随机打乱 candidate score 与 candidate return 的对应关系 100 次，得到 pairwise accuracy 均值 0.138、Kendall-tau-b 均值约 -0.001、top-1 regret 均值 `6.04e-4`。因此 0.104 不是“反向排序”的充分证据，主要反映模型把大量有效对预测为平局；但连训练集都未优于 zero predictor，表明下一步应优先排查 target 尺度、action/label 对齐和优化，而不是再延长训练或将责任归因于标签无效。

离线产物：

```text
/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/
influence_offline/eventvalue_sidecars.pt
/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/
influence_offline/report.json
```

## 8.2 新增诊断：重复动作噪声、小样本拟合与在线前端

### 重复动作标签审计

在同一批 160 个 clone state 中，candidate 0 与 candidate 1 是逐元素完全相同的完整 5-step action chunk。它们的即时环境分支结果一致：reward、实际持续步数、termination 和 truncation 的差异均为零；但 `R + gamma Vbar_E(endpoint)` 仍有差异。该差异完全来自 endpoint Event-Value bootstrap：中位 absolute difference 为 `2.18e-5`、P90 为 `1.30e-4`、最大为 `2.18e-3`。因此这不是物理动作差异，而是 endpoint 渲染/在线前端/Event Value 的非确定性或敏感性；后续 branch label 必须显式报告并超过这一噪声地板，不能仅凭 MSE 判断 Influence。

### 高信号小样本过拟合检查

新增 `--overfit-all-states`、action-repeat noise filter 与 target-scale-normalized loss。筛出 candidate range 大于 `max(3 × repeat-noise, 1e-4)` 的 92 个 state 后，以冻结 representation/action/return 训练 500 epoch（不保留 validation；这是纯可学习性诊断而不是泛化实验）。即使取消 weight decay 并按训练 target 标准差 `1.695e-3` 缩放梯度，结果仍为：

| 指标 | 高信号训练集（92 states） |
|---|---:|
| model / zero centered-MSE | `0.931` |
| tie-aware pairwise accuracy | `0.143` |
| Kendall-\(\tau_b\) | `0.071` |
| predicted ties | 318 / 414 eligible pairs |
| statewise shuffle pairwise accuracy | `0.097` |
| top-1 regret（model / shuffle） | `6.78e-4` / `1.35e-3` |

模型相对 zero predictor 仅有微弱 MSE 改善、仍把多数候选预测为平局。它略优于 statewise shuffle，但**尚未通过“训练集本身能可靠排序”的门槛**。该结果优先指向 action/target 尺度、action-label 对齐或当前 Influence MSE 目标的可学习性问题；它不证明 branch 标签无效。

### 在线前端端到端检查

新增 `evaluate_adjust_bottle_online_frontend.py`，在与 Observer 原验证相同的 15 条 held-out episode、256 个有效 5-step chunk 上，严格比较同一个冻结 Observer 的两种输入：cache teacher feature 与 `RGB student -> feature`。输入合同一致：`proprio_time_delta=5 control steps`、mount token `2`、右腕 RGB。

| 指标 | cache teacher \(\rightarrow\) Observer | RGB student \(\rightarrow\) Observer |
|---|---:|---:|
| boundary F1 | 0.7030 | 0.6951 |
| boundary F1@±3 | 0.9576 | 0.9512 |
| progress MAE | 0.1710 | 0.1718 |
| Event Value MAE | 0.1003 | 0.1090 |

在线路径相对 teacher 有小幅退化（teacher-to-online Value absolute MAE `0.0209`），但不能解释 Influence 在训练集上也难以排序。因此，当前主要排查对象仍是 branch target/Influence 学习，而不是 RGB student。

### 固定 SFT continuation 验证链路

新增 `validate_bootstrap_with_sft_continuation.py`：从真实 live RoboTwin snapshot 执行完整 5-step candidate chunk，计算

\[
R_{\mathrm{branch}}+\gamma\bar V_E(z_{\mathrm{endpoint}}),
\]

然后恢复相同 endpoint，由**冻结的** \(\pi_{0.5}\) SFT 连续重规划至终止，计算

\[
R_{\mathrm{branch}}+\gamma G_{\mathrm{continuation}}.
\]

首个 smoke state（25 control steps、2 candidate）成功执行和恢复：bootstrap 偏好 candidate 1（`0.942379 > 0.941706`），真实 SFT continuation 则偏好 candidate 0（`0.770043 > 0.762343`），即该单对排序相反。该单对不能作为 Value 无效的结论；它只证明新的直接标签校验已可运行。

随后完成时间分层运行：3 个 state（25/75/125 control steps）\(\times\) 3 candidates \(\times\) 每个 endpoint 2 次 SFT continuation。在 8 个真实 continuation 非平局 pair 中，bootstrap 排序 concordant/discordant 为 `7/1`，pairwise accuracy `0.875`；每 endpoint continuation 标准差为 `0–7.66e-3`，与部分候选间 return 差异同量级。该结果说明当前 \(\bar V_E\) bootstrap **并非显然没有动作排序信号**，但样本只来自一条 time-stratified SFT 轨迹、存在 continuation 随机性，不能用作 Influence 或最终方法有效的证据。后续必须扩大独立 reset seed、接触/调整/临近终止状态覆盖，并报告 repeat-noise-normalized 排序。

## 8.3 重复标签、状态内目标与 snapshot 诊断（最新）

### 重复候选合并

原始 160 个 state 的 candidate 0/1 均为逐元素相同的 5-step action chunk。确定性 Influence 不可能为同一个 \((z,a)\) 拟合两个不同 target，因此训练工具现在将每个 state 内 action 相同（由 `max|a_i-a_j|=0` 判定）的 candidate 当作重复测量：先平均其 branch return，再在合并后的唯一候选上重算 state mean 和中心化 target。所有 160 个 state 都从 4 个候选合并为 3 个唯一 action；使用 `range(Y)>max(3×repeat-noise,10^{-4})` 后保留 87 个高信号 state。

真实标签的 tie threshold 与预测 score 的 tie threshold 已分离：前者为 `1e-4`（并由 repeat-noise filter 进一步约束），后者在排序诊断设为 0，因此正数缩放不会伪造“预测平局”或改变排序结论。动作并非数值上几乎一致：合并后三候选 action-pair flattened \(L_2\) 的中位数为 `0.0503`、P90 为 `0.7784`。

### 同一网络、两种损失的可学习性比较

不改网络，均使用相同冻结 state representation/action、500 epoch、`lr=1e-3`、无 weight decay、训练 target std 归一化。结果中的 raw MSE 很大是预期的：网络可带有 state-common score offset；排序和 `state_centered_model_mse` 才是该诊断的可比量。

| 目标 | 高信号 overfit：pairwise / \(\tau_b\) | group-held-out（seed 49，20 states）：pairwise / \(\tau_b\) | held-out shuffle pairwise |
|---|---:|---:|---:|
| state-centered MSE | 0.970 / 0.886 | **0.774 / 0.514** | 0.512 |
| reliable-pair Huber | 0.953 / 0.853 | 0.755 / 0.479 | 0.497 |

两种目标都能学到显著高于 statewise shuffle 的 held-out 排序；当前小数据下 state-centered MSE 略好，故它是后续不改架构时的首选。注意 validation 只来自一个 reset-seed group，53 个非标签平局 pair，不可作为最终泛化或 PPO 开启门槛；它只推翻了“Influence 连训练/held-out 排序都学不会”的旧判断。

随后以四个现有 reset-seed group 做 leave-one-episode-group-out（LOEO）复核，state-centered MSE 的 held-out pairwise 为：seed 49 `0.774`、57 `0.600`、852 `0.481`、978 `0.768`；对应随机 score shuffle 约 `0.49–0.51`。seed 852 的 27 个高信号 state 仍近随机，故此前 seed-49 的结果不能概括为稳健泛化。当前数据只含 4 个相关 episode group，下一轮必须扩展独立 reset seed 和关键状态覆盖；不得因某一个 split 的 0.774 开启 PPO。

### 非零动作 snapshot repeat control

为定位重复 label，`validate_bootstrap_with_sft_continuation.py` 新增了完全重复 action control、endpoint 分量 hash，以及对**同一份 endpoint 输入**连续两次 sidecar 推理的检查。结果为：

* 同一 endpoint RGB/proprio 输入重复推理的 Event Value 差为 0，说明 sidecar 在 `eval()` 下是确定的；
* 同一 root snapshot、同一 nonzero 5-step action 的两次分支，其 endpoint head RGB、right-wrist RGB 和 `measured_state16` hash 均不同；bootstrap Value 差 `6.32e-4`；
* 独立的 snapshot validator 也以固定非零 action 复现了该现象：恢复后 reward/done/counter 一致，但 14/16 个 measured-state 元素有差异，最大 absolute difference `6.08e-6`。

所以端点输入差异不是 Influence 或 sidecar 推理随机性，而是当前 RoboTwin/SAPIEN snapshot restore 后非零控制的细微物理/控制器状态差异。此时不能把 raw branch return 当无噪声 counterfactual 标签。短期安全处理是：重复 action 合并、按 repeat-noise 筛除 pair、保存 repeat variance 并在后续收集时按其降权；长期需要继续审计 RobotWin task/controller 的未快照 Python state，或采用官方支持的更严格 state clone API。`lambda` 仍保持 0。

### 8.4 在线分数定义与下一轮顺序偏差诊断（待 A6000 运行验证）

state-centered MSE 只约束同一状态内的候选差异。若网络输出为 (f(z,a))，任意状态函数 (c(z)) 都可构成同样等价的分数 (f(z,a)+c(z))。因此不能把 raw (f(z_t,a_t)) 直接当作跨时间步 Influence 写入 PPO；event 内再中心化也不能消除每个状态各自不同的偏移。

代码现已改为在 rollout worker 从**同一观测**额外采样至少两个未执行的 Flow-SDE action chunk（仅增加模型推理，不增加 simulator interaction），并让 actor 使用：

\[
\widehat I(z,a)=f(z,a)-\frac1M\sum_{m=1}^{M}f(z,\tilde a_m),
\qquad \tilde a_m\sim\pi(\cdot\mid z).
\]

branch 的在线训练 loss 也已同步改为 candidate-score 与 candidate return 均作同状态中心化，避免把 raw score 回归到中心化 target。只有 `granularity=chunk`、`reference_candidates>=2` 且未来显式通过 ranking gate 时才允许非零 Event mixing；目前配置仍为 `max_lambda=0`。

续跑验证器新增 `--candidate-execution-order`。它允许以 `0,1,2` 和 `2,1,0` 等排列从同一 root snapshot 执行候选，但总是按 candidate identity 写出结果与 `execution_position`。下一次小规模 A6000 采集应比较同一 action 在不同执行位置的 endpoint 输入、bootstrap 与 continuation return：若差异系统性随 position 改变，则重复均值不能消除偏差，必须继续修复 snapshot/控制器状态；若主要是无方向随机波动，才能使用重复均值和方差构造可靠候选对。

## 9. 对当前方法效果的严格结论

截至本报告：

1. **除 snapshot 的严格物理重复性外，接口正确性得到较强支持。** chunk 时间单位、独立 Value target、GAE 隔离、sidecar 存档与单环境真实 RoboTwin rollout 均已经跑通；非零动作的 RoboTwin snapshot/restore 仍有可测 endpoint 偏差，不能再称为 exact same-state intervention。
2. **Observer/RGB student 在修复后 cache 上已完成任务匹配训练。** 这不等价于泛化能力或在线端到端成功。
3. **真实 candidate branches 有微弱但可测的 return spread。** 相比重复动作差异，平均候选差异约高一个数量级；但仍有不少 near-tie state，且 branch target 主要由短期 bootstrap 构成。
4. **Influence 的小样本可学习性已有正面证据，但泛化不稳定。** 经重复 action 合并、可靠 pair 筛选和 state-centered 目标后，训练集 pairwise 为 `0.970`；LOEO held-out 分别为 `0.774/0.600/0.481/0.768`。这支持“该组修改后的方案能学习部分未见 episode 的候选排序”，但 seed 852 近随机，且仅四个相关 episode group，不能把它写成稳健 held-out 结论。
5. **没有 EventValue-RL 成功率提升可报告。** 没有启动 Event-PPO，也不能把任何 `success_once` 当作 Ours 的性能。

因此，当前唯一严谨的实验状态是：

\[
\boxed{\text{Event branch pipeline works; Influence has partial held-out ranking evidence, but robust generalization, branch-label integrity, and PPO gain remain unproven.}}
\]

## 10. 当前代码改动

本轮相关本地提交：

| commit | 内容 |
|---|---|
| `79677277` | 对齐 chunk-strided Event cache mask |
| `f51dffc3` | 拒绝未对齐 cache 字段 |
| `7468632b` | cache 保留 terminal frame |
| `68d2f7cd` | 修复 cache 时间轴验证 |
| `956c9c93` | 拒绝 RGB source 不完整的 cache |
| `d27b4cd6` | 验证每一个 cache RGB boundary frame |
| `3fe8af10` | Influence 使用全部 matched branch candidates，并保存 Event representation |
| `e6f7b1e1` | 新增 grouped offline Influence 训练/验证工具 |

运行脚本 `examples/embodiment/run_robotwin_event_branch_a6000_smoke.sh` 也补充：可覆盖 checkpoint/output 路径、action chunk、branch interval、sidecar 学习开关，并显式提供 Curobo/PATH 给 A6000 Ray worker。

用户远端 `user/event-smdp-credit` 当前因 GitHub SSH 临时关闭连接而**尚未推送**本轮本地 commits；远端恢复后应执行：

```bash
git push user event-smdp-credit
```

本地工作区还存在用户原有的未提交 docs 变动；本轮没有覆盖或纳入提交。

## 11. 推荐后续顺序

不要直接扩训练或开启 `lambda>0`。优先级应为：

1. **A6000 接入回归。** 在 `lambda=0` 下验证新 transport 的 reference action shape、同状态相对 score 及 loss 均为有限数；再显式断言没有 reference action 时，任何未来的 `max_lambda>0` 配置立即拒绝启动。
2. **执行顺序偏差。** 用新 `--candidate-execution-order` 以至少两个相反排列执行同一批 root states；逐 action 对齐比较 endpoint、bootstrap 与 continuation。先区分位置偏差与随机 repeat noise。
3. **扩大独立 episode 与 continuation 验证。** 预先划分新的 train/validation/test reset seed groups，在接触、调整和临近终止状态上比较 `R_branch + gamma V_E` 与固定 SFT 的真实续跑回报；把续跑方差接近候选差异的 pair 标为不确定或增加重复次数。
4. **冻结标签生成器后再扩 Influence 数据。** 固定 Observer、RGB student 和 target Event Value，保留全部 state 与 high-signal 子集的比例，报告 LOEO、Kendall-\(\tau_b\)、top-1 regret、prediction spread 与 statewise shuffle；不要只在筛选后的容易状态上作总论。
5. **通过预注册门槛后才开始 PPO 混合。** 至少预先定义 held-out pairwise、Kendall-\(\tau_b\)、top-1 regret 相对随机/zero control 的门槛；之后以 `lambda: 0 → 0.1 → 0.25` 的保守 schedule 做 paired GAE 对照。原 PPO critic target 始终保持 GAE。
7. **完成 oracle 诊断阶梯。**

   \[
   \text{GAE}
   \rightarrow \text{oracle boundary + Event-SMDP uniform}
   \rightarrow \text{oracle event + true branch ranking}
   \rightarrow \text{predicted event + true ranking}
   \rightarrow \text{oracle event + learned Influence}
   \rightarrow \text{predicted event + learned Influence}.
   \]

   并加入 shuffle/reverse Influence falsification。只有真实 Influence 显著优于它们，才能作因果 credit claim。

## 12. 可复现路径索引

| 用途 | 路径 |
|---|---|
| A6000 测试代码 | `/home/chefmate/Data/pirl_a6000_run/RLinf-piRL-event-effect-test` |
| π₀.₅ RoboTwin SFT | `/home/chefmate/Data/pirl_a6000_run/RLinf-Pi05-RoboTwin-SFT-adjust_bottle` |
| verified cache | `/home/chefmate/Data/pirl_a6000_run/data/robotwin_adjust_bottle_event_cache_chunk5_v2_effect_verified2` |
| Observer | `/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/observer_chunk5/best.pt` |
| RGB student | `/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/rgb_student_chunk5/best.pt` |
| all-candidate calibration raw branches | `/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/calibration_all_candidates_raw` |
| offline Influence report | `/home/chefmate/Data/pirl_a6000_run/outputs/event_effect_verified2/influence_offline/report.json` |
| A6000 runtime log | `/home/chefmate/Data/pirl_a6000_run/logs/event_chunk5_calibration_all_candidates.log` |
