# RoboTwin `adjust_bottle`：πRL 与 EventValue-RL 实验完整记录

> 更新日期：2026-09-21
> 状态：**训练链路、感知预训练链路和独立评测链路均已跑通；当前完整 EventValue-RL 的效果实验失败，不能据此声称优于 πRL。**

## 1. 目标与结论

本实验以官方 π₀.₅ SFT 策略为初始化，在 RoboTwin 2.0 的单任务 `adjust_bottle` 上复现 πRL 风格的 Flow-SDE + PPO 在线后训练，并将原本 action-level GAE 的信用分配替换为事件级的 Event-SMDP value 与受控分支干预（intervention）信用分配。设计目标不是让事件分类本身提高成功率，而是检验以下命题：

\[
\text{事件价值提供长程时间抽象，受控 action intervention 找到真正改变未来事件价值的动作，}
\]
\[
\text{从而为 Flow-VLA 提供比按时间传播的 GAE 更准确、更省样本的 credit assignment。}
\]

这次第一轮实验已证明“代码可以训练”，但尚未证明上述科学命题：

| 方法/检查点 | 独立固定重置评测 | 结果 |
|---|---:|---:|
| π₀.₅ SFT（未做 RL） | 64 episodes | **40/64 = 62.5%** |
| 官方式 πRL Flow-SDE + PPO + GAE，step 25 | 64 episodes | **24/64 = 37.5%** |
| EventValue-RL，step 25 | 64 episodes | **0/64 = 0%** |
| EventValue-RL，step 92 | 64 episodes | **0/64 = 0%** |

因此，当前 EventValue-RL 的两条 0% 曲线是策略退化，不是“尚未充分评测”或“评测程序没读到权重”。同一评测器能得到 SFT 的 62.5% 和 GAE 的 37.5%，故评测管线本身有效。当前结果应作为失败/诊断实验保留，不能写成方法优势。

## 2. 任务、初始化与公平比较原则

### 2.1 当前主任务

- **平台**：RoboTwin 2.0，经 RLinf 的 RoboTwin PPO 集成运行。
- **任务**：`adjust_bottle`，单目标、多阶段操作任务。
- **actor 初始化**：官方 π₀.₅ RoboTwin `adjust_bottle` SFT 权重。
- **RL 底座**：RLinf 的 π₀.₅、Flow-SDE action sampling、PPO actor/critic 更新。
- **基线**：完全保留同一 actor、同一 SFT、同一环境、同一 Flow-SDE/PPO 配置，使用官方 action-level GAE。

本任务适合首先验证机制闭环，但它本身不足以支持“通用 long-horizon VLA”的广泛结论。后续若机制修复成功，至少还需增加 1--2 个不同 RoboTwin 操作原型（如双臂 handover、tool use 或 articulation），再补 CALVIN 或 LIBERO-Long 的外部长期序列验证。

### 2.2 交互预算纪律

主比较必须使 baseline 与 Ours 使用相同的：初始 SFT、环境交互总数、随机种子、PPO/Flow 超参数、评测协议和计算资源。对于 Ours，分支 rollout 也是 simulator interaction，必须计入总预算：

\[
N_{\rm total}=N_{\rm actor\ rollout}+N_{\rm branch\ rollout}。
\]

只有 credit estimator 可改变；不能因 Ours 使用额外 branch 数据却仍声称相同 sample budget。

## 3. 最终系统设计

### 3.1 总体数据流

```text
任务指令 / 角色定义
        ↓
离线 RobotWin RGB 回放 + 实测 articulation qpos
        ↓
SAM 3.1 teacher：首帧角色锚定、视频传播、必要时重锚定
        ↓
版本化 role-track / 低维关系特征缓存（Zarr）
        ↓
Role-Graph Event Observer F_E（离线预训练）
        ↓
z_t^E, event posterior, boundary b_t, progress p_t, uncertainty u_t
        ↓                                      ↘
在线 π₀.₅ + Flow-SDE + PPO rollout              Event Value V_E^π
        ↓                                      ↗
同 simulator state 的多 action branch → future Event Value 差异
        ↓
Influence Model I_ξ(s_t,a_t,z_t^E)
        ↓
signed Event-SMDP / intervention credit → PPO policy update
```

这里的因果证据不来自 GNN 边本身；图只提供关系表征。真正的干预证据来自“保存同一 simulator state 后，以不同 Flow-SDE action 继续短 rollout”的受控比较。

### 3.2 角色图与可部署感知协议

长期目标是 task-general Role-Graph Event Observer，而不是固定的 `object-target-gripper` 阶段分类器。角色集合可包含 `manipulated_object`、`goal_region`、`articulated_part`、`handle_or_actuator`、`tool` 和 `gripper`。角色对形成动态图，节点输入包括角色轨迹、可见性、置信度/缺失标志、ROI 外观；边输入包括归一化图像位置/尺度/速度、overlap、相对运动与可靠性。

输入严格限制为真机部署时可获得的信息：

- RGB 图像；
- replay 时从 articulation 读取的**实测**夹爪 qpos、归一化开合 \(\hat w(t)\)、因果导数、plateau 标志；
- 实测末端位姿、线速度和角速度；
- SAM 3.1 离线产生的 teacher role tracks 与其训练出的轻量在线前端/Observer。

深度、oracle segmentation、物体真实位姿、接触真值不进入 Observer 特征。它们只用于生成训练标签和评测。这样才能让训练分布包含 SAM 的真实失败模式，例如遮挡、ID switch、重定位失败和置信度下降。在线 PPO 不运行 SAM，也不查询离线 Zarr；缓存仅用于训练轻量的、可在线从当前 RGB 推理的 Student/Observer。

当前实现已含 RGB Student：它以当前图像/proprio 为输入，蒸馏离线 teacher 的 30 维关系特征；Event Observer 和 Student 在在线时冻结，Event Value 继续随当前 policy 的 rollout 更新。

### 3.3 Observer 输出和监督

Observer 不以“这是第几个任务阶段”作为唯一目标，而是学习跨任务复用的关系原语，如：

`approaching`、`contact`、`attached`、`co-moving`、`constrained-motion`、`aligned`、`inside/on-target`、`released`、`articulating`、`actuated`。

这些原语经图网络融合成：事件状态 \(z_t^E\)、event posterior、boundary probability \(b_t\)、progress \(p_t\) 和 uncertainty \(u_t\)。几何可得的原语与状态/外观变化主导的 `actuated` 应分 head 并分开报告，避免把“按钮/开关外观变化难判断”掩盖进平均 F1。

### 3.4 Event-SMDP Value

对于事件 \(E_j=[\tau_j,\tau_{j+1})\)，持续时间 \(D_j=\tau_{j+1}-\tau_j\)，使用事件半马尔可夫 bootstrap：

\[
Y_j=R_j+\gamma^{D_j}V_E(z^E_{j+1}),\qquad
A^E_j=Y_j-V_E(z^E_j)。
\]

其中 \(R_j\) 是事件期间的折扣累计 reward。离线 expert/SFT 轨迹用于初始化 \(V_E\)，而在线阶段 \(V_E^\pi\) 必须继续使用当前 policy 的 event return 更新；否则策略改变后 value 会严重 OOD。

### 3.5 受控干预与 signed credit

在少量关键/边界候选状态保存 simulator state。从完全相同的状态，对当前 action 和约 4 个替代 Flow-SDE action 分支作 10--20 step 短 rollout：

\[
I(s,a)=Y(s,a)-\mathbb E_{a'}Y(s,a')，
\quad Y(s,a)=R_{t:t+H}+\gamma^H V_E(z^E_{t+H})。
\]

因为不能在每个 timestep 都实际 clone，真实 branch 样本监督轻量 Influence Model \(I_\xi(s,a,z^E)\)，由它给 event 内所有 step 预测影响。当前实现的分配是 signed、按绝对影响归一：

\[
w_t=\frac{|I_\xi(t)|}{\sum_{k\in E_j}|I_\xi(k)|+\epsilon},\qquad
A_t^{\rm EVI}=D_jw_t\operatorname{sgn}(I_\xi(t))|A_j^E|。
\]

这个公式保留“有害 action 应得到负 credit”的信息，优于 softmax 权重；但也意味着 influence 的符号一旦错误，就会把好动作主动惩罚，是当前失败的核心风险。

## 4. 已完成的数据与预训练

### 4.1 轨迹和缓存

本轮没有使用跨 benchmark 的 RobotWin 数据来给 ManiSkill/CALVIN 对比提供额外信息；数据、Observer 和 RL 均闭环在同一个 RoboTwin `adjust_bottle` 任务内。

- 原始 task-matched trajectory collection：**410** 条；
- SAM audit：抽检 **54/54** 通过；
- 最终可用于正式预训练的有效 SAM cache trajectory：**207** 条；
- SAM teacher 输出：角色 mask/box/track、置信度、可见性、轨迹哈希及低维关系特征；
- oracle 仅提供接触、抓持、抬升/移动、释放、成功、event boundary 等监督与评估标签。

该量级可以作为 `adjust_bottle` 的第一版 task-matched 预训练，但不足以声称大规模多 archetype 表征泛化。未来需在训练 benchmark 内为每个任务独立构建类似缓存；RobotWin 多任务缓存则适合作为附录中的跨原型 Observer 实验。

### 4.2 预训练结果

Event Observer + Event Value 预训练在有效 cache 上完成，最佳 epoch 为 35：

| 指标 | 最佳验证结果 |
|---|---:|
| boundary accuracy | 0.9305 |
| progress MAE | 0.1458 |
| event-value MAE | 0.0546 |

RGB Student 单独蒸馏 30 维 teacher relation feature，训练 30 epochs，最佳验证 teacher-feature MAE 为 **0.0135**（约 epoch 27）。

这些结果只能说明离线标签上的事件表示和 teacher feature 蒸馏可学；它们**不等价于**在线 policy distribution 下的 event boundary 正确，也不保证 \(V_E\) 或 \(I_\xi\) 的 action-level credit 正确。

预训练 checkpoint：

```text
/data/user/leviccdong/EKSF/outputs/
  robotwin_adjust_bottle_event_pretrain_v1_200plus/best.pt

/data/user/leviccdong/EKSF/outputs/
  robotwin_adjust_bottle_rgb_student_v1_200plus/best.pt
```

## 5. 在线训练如何运行

### 5.1 官方 πRL 基线

基线作业 `637047` 已正常结束（exit code 0），运行 100 个训练 step/epoch。它采用：

1. π₀.₅ SFT actor 按原 RLinf 路径输入 RGB/proprio；
2. Flow-SDE 采样 action chunk，并在 RoboTwin 执行；
3. 原 actor critic 以 action-level GAE 计算 advantage；
4. 标准 PPO clipped objective 更新 actor/critic。

完整 baseline checkpoint 位于：

```text
/data/user/leviccdong/EKSF/outputs/robotwin_pi05_gae/
robotwin_adjust_bottle_pi05_flow_sde_action_gae/
checkpoints/global_step_100/.../full_weights.pt
```

### 5.2 EventValue-RL 训练

正式 EventValue-RL 作业 `637515` 正常结束（exit code 0），共完成 92 个训练 step/epoch，耗时约 1 小时 14 分。其每轮工作如下：

1. 当前 π₀.₅ actor 的 Flow-SDE rollout 产生正常 PPO batch；
2. 冻结的 RGB Student + Observer 从实时 RGB/proprio 产生 \(z^E\)、事件 ID/boundary/progress；
3. 在线更新 \(V_E^\pi\)；
4. 周期性选择 action chunk 候选点 clone simulator state，采样 4 个替代 action branch、作短 horizon rollout；
5. 用 branch 目标训练 \(I_\xi\)；
6. 以 Event-SMDP + signed influence advantage 替代 action GAE，执行同一 PPO update。

训练没有崩溃：Flow-SDE rollout、branch、Event Value loss、Influence loss、actor backward、权重同步和 checkpoint 均实际执行。最终 checkpoint 与 sidecar：

```text
/data/user/leviccdong/EKSF/outputs/robotwin_eventvalue_200plus_formal/
robotwin_adjust_bottle_eventvalue_200plus_formal/
checkpoints/global_step_92/actor/model_state_dict/full_weights.pt

.../checkpoints/global_step_92/actor/eventvalue_sidecars.pt
```

sidecar 内实际保存 Observer、RGB Student、Event Value、Influence Model、Influence optimizer 与 branch 计数；step 25/50/75/92 时的累计真实 branch supervision 分别仅为 **20 / 40 / 60 / 74**。

## 6. 评测设计与最终结果

训练内的低样本 eval 曾报告 Event step 25 为 6.25%，随后 step 50/75/92 为 0；这类 16 episode 中途评测方差很大，不能作为最终结论。因此新增了独立、确定性、固定重置的 **64 episode** checkpoint evaluator。它不加载 Event sidecar 来采样动作，只读取各 checkpoint 的 actor 权重；这是正确的，因为 sidecar 只改变训练时的 credit，不应改变测试 policy 的动作接口。

最终结果见第 1 节表格。两个关键事实是：

1. SFT 和 GAE checkpoint 在同一 evaluator 下均有非零结果，排除了“相机、环境、权重路径或 evaluator 全部坏掉”的解释；
2. Event step 25 和 step 92 都是 0，表明不是单纯训练太短，而是 Event credit 更新后策略持续处于退化状态。

额外观察：GAE baseline 的 37.5% 也低于 SFT 的 62.5%，说明当前单 seed、短预算的 PPO 配置本身已经有明显方差/过更新风险；但是 Event 的 0% 明显更差，不能归咎于 baseline 不稳定。

## 7. 失败归因

目前最有证据支持的原因如下。

### 7.1 过早完全替换 GAE

当前训练一开始就以 Event-SMDP/signed influence 取代原始 GAE。在线早期成功回报很少，日志后期出现 `reward=0`、`success_once=0`，原 critic explained variance 也为 **0**。此时 \(V_E\) 的 bootstrap target 近乎全零，event advantage 不能提供稳定学习信号。

### 7.2 Branch 监督严重不足且被外推到所有动作

step 92 只有 74 条真实 branch label。当前 action chunk 长度约为 50，branch 又是按周期性 chunk 候选点而非精确事件关键 timestep 获取；但 \(I_\xi\) 被要求外推至每个 event 内所有 action。signed credit 使任何影响符号误判都会由“少给正奖励”升级为“主动给负 advantage”，容易迅速破坏 SFT 行为。

### 7.3 离线监督与在线分布错配

Observer 的 boundary accuracy 0.9305 是离线 cache 标签上的总体准确率，不代表在线失败/遮挡状态下 boundary 定位准确。更重要的是，Event Value 在 expert/SFT distribution 初始化后，随着 PPO actor 改变会遇到 OOD 状态；若同时持续看到零成功 episode，会形成“低 value → 无有效 credit → 更差 policy”的反馈回路。

### 7.4 时间粒度不匹配

当前事件表示/branch 主要绑定在 policy chunk 级，而要分配 credit 的对象是 chunk 内低层 action timestep。事件边界、干预状态和实际造成结果变化的动作尚未严格对齐。因此当前结果只说明该第一版 action-level intervention 编码不稳定，不能反证“事件价值思想”本身。

## 8. 已验证、未验证与不能宣称的内容

### 已验证

- 官方 π₀.₅ + RLinf RoboTwin Flow-SDE/PPO/GAE 从 SFT 初始化可端到端运行；
- Task-matched RobotWin replay、oracle label 对齐、SAM teacher cache、Observer/Event Value/RGB Student 预训练均可运行；
- Event Value sidecar、simulator cloning/branch rollout、Influence Model、signed credit 与 PPO 接口均被真正调用；
- 独立 checkpoint evaluator 可稳定加载和评测 SFT、GAE、Event checkpoint。

### 尚未验证

- Event-SMDP 相较 GAE 是否提高 critic explained variance、success-vs-interaction AUC 或达到目标成功率所需交互数；
- Influence Model 是否能正确预测真实 branch influence 的大小、排序和符号；
- Observer 在 online policy distribution、其他任务/物体/场景上的泛化；
- 任意关于 sample efficiency、long-horizon superiority、sim-to-real 或跨 benchmark 的结论。

## 9. 下一版必须实施的修复和消融

### 9.1 安全的 credit 迁移

不再硬替换 GAE。先使用残差混合和 warm-up：

\[
A_t=(1-\lambda)A_t^{\rm GAE}+\lambda A_t^{\rm EVI},
\]

初始 \(\lambda=0\)，在 Influence Model 的 held-out 真实 branch 排序/符号准确度通过阈值后逐渐增加。早期可先使用 Event-SMDP uniform allocation，再启用 temporal/boundary credit，最后才启用 signed intervention credit。

### 9.2 先收集、再使用 branch influence

冻结或极小更新 actor，先在接近 SFT 的 policy 下积累至少数千条 branch supervision；对 held-out branch 报告：

- 未来 Event Value 预测 MAE；
- influence 符号 accuracy；
- influence ranking Kendall-\(\tau\)/Spearman；
- 关键 timestep ranking accuracy。

只有这些指标可接受时，才让 \(I_\xi\) 参与完整 actor update。branch 状态应由高 boundary/高 uncertainty/高 predicted influence 选择，而不是主要按固定 chunk 周期选择；且所有 branch transition 计入交互预算。

### 9.3 最小、可解释的消融链

在相同总交互预算、至少 3 个 seed 下运行：

1. `π₀.₅ SFT`（只评测）；
2. `πRL Flow-SDE + PPO + action-GAE`；
3. `πRL + Observer + GAE`：控制额外网络/标签带来的影响；
4. `Event-SMDP + uniform credit`；
5. `Event-SMDP + temporal/boundary credit`；
6. `Full: Event-SMDP + calibrated branch-interventional credit`。

主指标为 success-vs-total-interaction AUC、达到相同 success 阈值的 \(N_{80}/N_{90}\)（任务可达到时）、最终成功率、跨 seed 均值/标准差，以及 credit ranking accuracy。对 `adjust_bottle`，若成功率 ceiling 已高，应同时报告“相同成功率所需总 simulator transitions”。

### 9.4 Observer 泛化实验与外部验证

在机制修复后，把 RoboTwin 扩展到 6 类操作 archetype、12--18 个具体任务。切分应优先使用 task/object/scene 分层及“原语已见但组合未见”的 archetype compositional holdout，而非仅 episode holdout。RoboTwin 负责多原型关系表征与 same-state intervention；CALVIN ABC→D 或 LIBERO-Long 负责真正连续长程任务的外部验证。CALVIN 中任何 SAM cache、oracle 标签、Observer/Value 预训练、PPO 和 branch 都只能来自 ABC，D 仅用于 zero-update evaluation。

## 10. 代码与产物定位

本地仓库：`/home/jj/RLinf-piRL`，分支：`event-smdp-credit`。本轮相关实现包括：

```text
rlinf/models/embodiment/event_observer.py
rlinf/algorithms/event_value.py
rlinf/algorithms/event_credit.py
rlinf/algorithms/event_intervention.py
rlinf/workers/actor/embodied_fsdp_actor_worker.py
examples/embodiment/slurm_robotwin_pi05_eval.sh
examples/embodiment/train_embodied_agent.py
rlinf/runners/embodied_runner.py
rlinf/scheduler/cluster/cluster.py
```

独立 eval runner 的关键提交依次为：`8843a5ec`、`623237d3`、`7efa6927`、`a4a833e2`、`f3134af3`。其目标是避免 eval 时错误初始化训练 actor/FSDP、避免 Ray dashboard 端口碰撞、补全 π₀.₅ rollout schema，并为各 checkpoint 隔离输出目录。相关单元测试已通过：event sidecar、event credit、event intervention 共 **11 passed**。

## 11. 一句话总结

当前已完成的是一条真实可运行的“SAM teacher → Role-Graph Event Observer → online Event Value → simulator branch intervention → PPO”的系统；当前未完成的是让这条系统在不破坏 π₀.₅ SFT 行为的前提下提供有效 credit。下一轮实验的首要目标不是扩大任务数，而是以 GAE safety residual、足量且经校准的 branch supervision、严格总交互预算和 3-seed 消融，先把 `adjust_bottle` 上的 Event credit 从 **0% 退化**修复为可重复的非退化/正向增益。
