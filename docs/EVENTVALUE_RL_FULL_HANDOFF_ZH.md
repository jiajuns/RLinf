# EventValue-RL / πRL 全量交接报告

> 用途：本文件是当前项目的可续接状态快照。新的 AI 或研究者应先完整阅读本文件，再提交、修改或解释任何实验。它将“已验证事实”“尚未验证的设计”“已经失败的尝试”严格分开，避免把链路打通误报成方法有效。
>
> 文档写入时间：2026-09-22（Asia/Shanghai）。其中 Slurm 作业时间戳均按原始日志保留（主要为 2026-09-21）；“当前 HPC 调度快照”以本次写入前的最后一次查询为准，续接时必须重新查询。

---

## 0. 一句话状态

研究目标是将 **Event-SMDP Value + matched-state branch intervention credit** 接入官方 **π₀.₅ + πRL Flow-SDE/PPO**；当前代码已实现 V2 的保守 residual-credit 版本，RoboTwin `adjust_bottle` 上的 V1 曾使成功率降到 0%，因此尚无方法正增益。V2 已完成一次 4-GPU branch 链路验证（128 个 branch state、512 个 branch outcome），但计划中的续采集作业 `639000` 在首轮 rollout 前被 Slurm 取消，尚未达到 V2 actor 更新要求的 500+ branch state，也尚未运行正式 V2 PPO 成功率测试。

不要得出“Event 方法有效”或“πRL 官方效果很差”的结论。

---

## 1. 论文要解决的问题与最终方法主张

### 1.1 研究问题

目标不是证明“事件识别有用”，而是验证：

\[
\boxed{
\text{事件价值提供长程时间抽象；同一 simulator state 下的受控 action intervention 测量真正改变未来事件价值的决策；二者共同为 flow-VLA 提供更准确、样本更省的 credit assignment。}
}
\]

唯一的 VLA/RL 底座应为官方 OpenPI **π₀.₅** 与 RLinf/**πRL** 的 Flow-SDE + PPO。不得自行重写 VLA、Flow likelihood 或通用 PPO 框架。基线保持相同 SFT 权重、相同任务、相同 Flow-SDE、相同 PPO loss/优化器、相同总 simulator interaction 预算；核心变量只能是 credit estimator。

### 1.2 完整计算图

\[
\begin{aligned}
&\text{benchmark 内部 expert/SFT rollout 的 RGB + deployment-available proprio + oracle labels}\\
&\downarrow\\
&\text{SAM 3.1 离线 teacher tracks / versioned Zarr cache}\\
&\downarrow\\
&\text{online-capable Role-Graph Event Observer }F_E\\
&\downarrow\\
&(z_t^E,\;q_t^{event},\;b_t,\;p_t,\;u_t)\\
&\downarrow\\
&\text{online-updated Event Value }V_E^\pi(z_t^E)\\
&\downarrow\\
&\text{sparse same-state Flow-SDE branch rollouts}\\
&\downarrow\\
&\text{Influence Model }I_\xi(s_t,a_t,z_t^E)\\
&\downarrow\\
&\text{credit-conserving Event-SMDP residual advantage}\\
&\downarrow\\
&\text{πRL PPO update of π₀.₅ actor.}
\end{aligned}
\]

### 1.3 三个模块及其边界

1. **Role-Graph Event Observer (F_E)**
   - 输入是部署时能取得的 RGB 与实测本体感知；输出统一 event representation、relation primitives、event posterior、boundary、progress 与 uncertainty。
   - 它不是“因果证据”来源。GNN/Observer 只表示状态和关系；真正的因果证据必须来自固定 simulator state 后施加不同 action 的 branch intervention。
   - 线上 PPO 不运行 SAM；SAM 是离线 teacher/cache 生成器。线上只能运行轻量角色前端 / RGB student + Observer。

2. **Event Value Critic (V_E)**
   - 事件 (E_j=[s_j,\ldots,s_j+D_j-1]) 的 SMDP target：
     \[
     G_j^E=R_j+\gamma^{D_j}V_E(z_{j+1}^E),\qquad
     A_j^E=G_j^E-V_E(z_j^E).
     \]
   - 它应先由同一 benchmark 的 expert/SFT 轨迹初始化，online 时继续按当前 policy rollout 更新；Observer 可冻结，而 value 不能整个 online 阶段冻结，因为 (V_E^\pi) 是 policy-dependent。
   - branch target 使用 EMA target critic，避免当前快速变化的 value 同时污染 event advantage 与 influence target：
     \[
     \bar V_E\leftarrow \tau\bar V_E+(1-\tau)V_E,\qquad
     Y=R_{t:t+H}+\gamma^H\operatorname{stopgrad}(\bar V_E(z_{t+H}^E)).
     \]

3. **Influence Model (I_\xi)**
   - 在保存的同一 simulator state 从 π₀.₅ Flow-SDE 采样约 (K=4) 个 action chunk；每支短 rollout (H\approx10\)--20 step。
   - branch 目标是每个 candidate 的未来 Event return (Y(s,a))，真实 influence 可写为：
     \[
     I(s,a)=Y(s,a)-\frac{1}{K}\sum_{a'}Y(s,a').
     \]
   - 真实 branch 只能稀疏地做，否则 interaction 成本爆炸。因此必须用这些标签训练 (I_\xi(s,a,z^E))，再为 event 内全部 chunk/time-step 推断 influence；不能把少量 branch 标签直接外推为每个低层动作的真值。

---

## 2. V1 为什么失败，以及 V2 的确定修复

### 2.1 已失败的 V1

V1 使用近似如下的 signed-absolute credit：

\[
A_t^{EVI}=D_j w_t\operatorname{sgn}(I_t)|A_j^E|.
\]

在只有约 74 条真实 branch supervision 时，Influence Model 的正负号直接决定 PPO 对 action 的奖励/惩罚方向。一次符号误判会把 SFT 的正确动作变成负 advantage，形成“policy 分布漂移 → influence OOD → 更错误 credit”的正反馈。

此外，π₀.₅ action chunk 长约 50，真实反事实实际回答的是“这个 action **chunk** 好不好”，不是 chunk 内第 17 个 low-level action 好不好。用 chunk-level counterfactual 给每个 action-level timestep 归因在识别上不适定。

### 2.2 V2 residual Event credit（当前代码中的目标版本）

V2 的原则是：**Event-SMDP 决定整个事件应得到多少总正/负 credit；Intervention 只在事件内部重新分配 credit，不得单独翻转整个策略更新方向。**

对一个 event 内的 predicted influence，定义中心化、截断后的相对量：

\[
\bar I_j=\frac1{D_j}\sum_{t\in E_j}I_t,
\qquad
c_t=\operatorname{clip}\left(
\frac{I_t-\bar I_j}{\operatorname{Std}(I_{E_j})+\epsilon},-c,c\right).
\]

V2 事件内 advantage：

\[
A_t^{Event}=\frac{A_j^E}{D_j}+\beta |A_j^E|c_t.
\]

最终交给 PPO 的 advantage 是 GAE residual mix：

\[
\boxed{
A_t=(1-\lambda_t)A_t^{GAE}+\lambda_t A_t^{Event}.
}
\]

重要性质是 
\(sum_{t\in E_j}A_t^{Event}\approx A_j^E\)。因此影响模型只把 credit 从低影响决策移动到高影响决策，不会借由不可靠符号把整个 event 的总更新方向倒置。`event_mix_lambda=0` 必须与 action/chunk GAE 严格等价，这也是重要 sanity check。

### 2.3 V2 启用门槛

在 actor 收到非零 Event residual 之前必须满足：

- 足够真实 matched-state branch state：最低暂定 **500**，更稳妥目标 500--1,000 state；每 state 4 candidates，对应 2,000--4,000 outcome；
- 在 held-out state 上报告 `pairwise ranking accuracy`、Kendall-\(\tau\)、sign accuracy，而非只报告 MSE/loss；
- Observer 在 SFT rollout、early πRL rollout、失败/recovery rollout 上报告 boundary Precision/Recall/F1、F1@\(\pm k\) steps、event F1、progress MAE；不能用 class-imbalanced boundary accuracy 替代；
- Event critic / influence 使用 EMA target，actor 初期保持 GAE 或很小的 \(\lambda\)，逐步如 \(0\to0.1\to0.25\)，第一版不应升至 1；
- branch interaction **必须计入总交互预算**。主表同时报告 total budget（actor rollout + branch rollout）和 actor-only budget。

当前 V2 采用 chunk-level credit，符合真实 intervention 粒度；不要先回退到 chunk 内 action-level counterfactual attribution。

---

## 3. Observer 设计：目标协议与当前实现状态

### 3.0 从旧 YOLOE+GNN 到当前 Role-Graph Observer 的架构演进

项目最初的实现路径是：冻结 YOLOE 仅从 RGB 输出 `object / target / gripper` 候选框、置信度和 ROI，不将 detector 输出当成关系/事件真值；将候选实体组成二维角色图，节点包含角色、框、置信度、轨迹、missing/ambiguity 特征，边包含相对位置、尺寸比、IoU、距离；两层 GNN 做帧内 object--target--gripper 关系传播，再融合 global image、ROI、最后有效帧表示与历史有效帧均值。旧 head 预测抓持、承托、目标区域、阶段、转移、短期关系和目标满足。CfC 已从这条路径删除，旧 CfC checkpoint/指标只能作历史对照。

这套“detector + graph encoder”不需要被概念上推翻，但其定位已改变：它只能是 **Event Observer 的角色/关系表征骨干**，而不是关系或因果事件真值，更不是论文的最终目标。后续经讨论，视觉角色生成器优先采用 SAM 3.1 offline teacher，因为目标是跨 archetype 可变角色图，而非仅 `object-target-gripper` 的 pick-place 阶段分类。

当前应采用以下兼容原则：

- 若现有 YOLOE+2-layer GNN 代码能输出角色轨迹、missing/reliability 与 pair relation，则可复用为轻量 online role frontend / graph backbone；
- 不再把旧 task-specific heads 当主研究结果；可作为 auxiliary supervision；
- 新的主输出是统一的 `z_t^E`、relation primitives、event posterior、boundary、progress、uncertainty；
- 真实在线 PPO 不应依赖逐帧 text prompt 或在线 SAM；SAM 负责离线 teacher 数据，轻量 student/role frontend 负责在线；
- “causal tracker”命名应停止使用，建议统一为 **Role-Graph Event Observer**；真正 causal evidence 的名称应保留给 matched-state intervention。

这意味着当前仓库中的 Observer/RGB student prototype 与最初 YOLOE code 并不是二选一关系；但尚未完成“可变角色、多 archetype、SAM teacher + online graph frontend”的完整统一重构，不能夸大为已完成。

### 3.1 最终感知协议（已确定的研究原则）

部署协议锁定为 **RGB + 部署可得的实测本体感知**：

- 不用 simulator oracle segmentation、object pose、depth、接触真值作为 Event Observer 输入；它们只用于离线监督与评估。理由是最终真机能取得关节/编码器、末端位姿、RGB，但通常不能取得 oracle segmentation/真值位姿。
- `w(t)`、末端 pose/线速度/角速度应是 articulation replay 后读到的 **measured qpos/state**；RoboTwin 官方 HDF 的 `joint_action/*` 与 `endpose/*` 是指令，不能伪装成 measured。
- 主方法不假定深度可用；若部署硬件的 RGBD 同步、外参、可用性将来被确认，可作为附加 ablation，不能混入当前主结果。
- 不做每 embodiment 手眼外参标定，也不固定“指尖像素原点”；不使用伪精确的 object--tip distance。腕相机自然是夹爪局部视角，安装差异由图像坐标归一化、随机 crop/translation augmentation、多 embodiment 训练和 camera-mount token 吸收。
- 夹爪开合输入用行程归一化的 measured \(\hat w\)、因果导数 \(d\hat w/dt\)、plateau flag；不用 action/commanded gripper state，防止 grasp 从 action 平凡泄漏。

RobotWin 数据优先使用官方 expert trajectory，而不是为了 Observer 重新生成同类 demonstration；但必须 replay episode 补齐 articulation 的 measured qpos/end-effector state、可用速度与 simulator oracle label。replay 的 oracle contact/grasp/lift/release/success/boundary 只作监督和评估。离线数据构建应按 episode 生成 manipulated object/goal/part 等角色 track，完整 episode 检查可见率、置信度、ID/box jump 与遮挡恢复，再将低维 RGB/proprio relation feature、mask/reliability 与 oracle label 对齐写入版本化 Zarr/cache。

### 3.2 离线 SAM 3.1 teacher 与线上 student

- object/target/part 角色轨迹应由真实运行的 **SAM 3.1** 生成：首帧 exemplar 或简单名词短语 grounding、视频 propagation、遮挡/ID switch/目标迟出现后的 re-anchor。
- SAM 必须真跑，不能直接拿 simulator oracle mask 造特征；否则 observer 没见过真实边界抖动、ID switch、遮挡重定位失败，而这些恰是 event boundary 的关键帧。
- SAM 结果写入带版本号与轨迹 hash 的离线 Zarr/cache；线上 πRL/PPO 不跑 SAM、也不能查询离线缓存（online rollout 会访问缓存中不存在的新状态）。
- 当前代码已训练一个 RGB student，使 online sidecar 可由 RGB/proprio 工作；SAM cache 是 teacher/training data，不是 PPO online lookup。

SAM 工程经历过一个必须保留的失败记录：最早在 RoboTwin 左腕第 0 帧用单次 `can` prompt 做 4-frame debug 时，SAM 没有返回初始候选，因而没有传播锚点、也没有产出 `.npz` mask/trajectory。该错误只证明“把一次首帧 grounding 写死”不稳，不是 SAM 权重损坏；随后离线 cache/audit 已能产生可用轨迹（54/54 audit）。完整 episode pipeline 仍必须支持 re-anchor，而不是依赖一次 text grounding。

### 3.3 Task-general Role-Graph 目标设计

固定 `object/target/gripper` 方案已被概念上升级为可变角色集：

`manipulated_object / goal_region / articulated_part / handle_or_actuator / tool / gripper`。

任务指令只用于首帧角色绑定；网络不接收 task ID/task embedding，真正学习跨任务共享 relation primitive：

`approaching, contact, attached, co-moving, constrained-motion, inside-or-on-target, aligned, articulating, actuated, released`。

建议将 head 分为：

- geometric relation heads：相对位置/速度、contact proxy、co-motion、overlap、alignment、inside/release；
- state-change heads：button pressed、drawer opened、switch/faucet turned 等外观/状态变化。

`actuated` 并非总是几何可得：按键位移很小、开关/出水可能是外观语义变化。论文中应单列该类指标和 limitation；不要把它藏在平均 event F1 中。

角色对边是可选的。例如 press button/turn faucet/open drawer 并不总有 goal_region；缺失边必须显式 `not-applicable/missing`，绝不能填零，否则模型把“没有 target”误学为“离 target 很近”。

### 3.4 推荐数据划分

episode-level holdout 只能用于快速 sanity check，不能用于泛化结论。应优先：

1. task-held-out：训练见过 primitive，但不见过具体任务；
2. object-held-out；
3. scene-held-out（材质、光照、干扰物）；
4. archetype-held-out compositional：测试任务中的 primitive 都在训练中出现过，但组合方式未出现。

不要将一个训练集中完全没有的 primitive（例如 constrained rotation）整类留给测试后，再把失败称作组合泛化失败。

### 3.5 当前 Observer 的真实状态

已完成的是 `adjust_bottle` 相关的离线 prototype，并非已完成的多 archetype 大规模 task-general Observer：

- 原始轨迹：410；有效 cache：207；
- SAM audit：54/54；
- best observer（epoch 35，cache/offline 指标）：boundary accuracy `0.9305`、progress MAE `0.1458`、event-value MAE `0.0546`；
- RGB student：MAE `0.0135`。

这些数值仅说明离线 cache 上可以拟合，不能表示 online usability 或 sim-to-real。尚缺 boundary P/R/F1、F1@\(\pm3\) frame、event primitive F1、online SFT/πRL failure rollout 评测、archetype/task/object/scene 泛化结果。

---

## 4. 平台与实验策略

### 4.1 主张的 benchmark 闭环原则

主论文的 πRL 对比不能出现“方法先看 RoboTwin 数据，baseline 没看”的数据不公平。对于每个主 benchmark，应独立闭环：

\[
D_B^{expert/SFT}\to\text{SAM teacher/cache}\to F_E^B,V_E^B\to\text{πRL vs Ours on }B.
\]

Observer 可在 benchmark 内 expert/SFT data 上预训练，online 后冻结；Event Value 必须随当前 policy 更新。RoboTwin 的多 archetype pretraining 可作为 representation/generalization 附录，但不应被混入另一个 benchmark 的主对比以制造额外数据优势。

### 4.2 平台职责

| 平台 | 合理职责 | 当前状态 |
|---|---|---|
| RoboTwin `adjust_bottle` | 首个 π₀.₅+PPO→Event-SMDP→branch credit 机制闭环；同 checkpoint 下的可控对照 | 已有 SFT、代码、branch debug；尚无 V2 PPO 正式结果 |
| RoboTwin 多任务（12--18 tasks / 6 archetypes） | Role-Graph Observer 的 multi-archetype 表征及组合泛化 | 规划中，未完成大规模任务通用预训练 |
| ManiSkill3 | `get_state_dict/set_state_dict` 式 same-state branch 机制验证；对齐 πRL 公开 benchmark | 早期 smoke/基线尝试存在；正式可靠复现尚未完成 |
| CALVIN ABC→D | 真正 sequential long-horizon 外部验证；A/B/C 所有训练，D 仅 zero-update eval | 官方 π₀.₅ SFT 已下载；当前有独立 CALVIN 作业在跑，尚未在本报告中验证其结果 |
| LIBERO-Long | 第二套独立长程验证 | 未完成 |

CALVIN ABC→D 的纪律：SAM cache、oracle labels、Observer pretrain、offline Event Value、online PPO、branch intervention 一律只来自 A/B/C；D 只允许最终 zero-update evaluation。D 上生成训练标签或 online update 都构成泄漏。

### 4.3 RoboTwin 任务选择

RoboTwin 可作为主机制平台，因为当前 RLinf 已支持 π₀/π₀.₅ + PPO 的 RoboTwin 配置。初期 `adjust_bottle` 最合适：已有单任务 SFT 与官方 PPO recipe。后续可扩展到多 archetype，如 `move_can_pot, place_can_basket, press_stapler, click_bell, open_microwave, turn_switch, beat_block_hammer, stack_blocks_two, handover_block, pick_dual_bottles`。

不要把 RoboTwin task 简单称为“πRL 论文全部跑过的任务”；它首先应承担 multi-stage/multi-archetype mechanism evidence。真正 long-horizon sequential claim 需要 CALVIN 或 LIBERO-Long 支撑。

---

## 5. 官方 πRL、SFT、PPO 的关系

- **π₀.₅ base**：通用的预训练 flow-VLA 参数，不等于任何 benchmark 可执行 policy。
- **SFT checkpoint**：针对 benchmark/task distribution 用 demonstration 监督微调得到的起点；例如 RoboTwin `adjust_bottle` checkpoint 只适用于该任务，不能代替 CALVIN/ManiSkill SFT。
- **PPO/RL**：从相同 SFT 初始化，用 simulator reward 做 online fine-tuning。
- **πRL**：一套面向 flow-VLA online RL fine-tuning 的方法/实现（Flow-SDE 或 Flow-Noise 的 action sampling、logprob、actor-critic/PPO 等），不是“一个能自动适用于所有任务的单 checkpoint”。必须匹配每个环境的 observation/action/norm stats 与 SFT 权重。

Flow-SDE 与 Flow-Noise 都是 πRL 支持的 flow perturbation/log-prob 路径。当前本项目主线锁定 **Flow-SDE**，以免在比较里同时改变 noise method 和 credit estimator。

---

## 6. 官方 RoboTwin π₀.₅ 配置与当前 4-GPU 对齐

官方 ready recipe 的关键特征：

- π₀.₅、Flow-SDE、PPO、chunk-level reward/logprob、GAE；
- action chunk 约 50，action dimension 14；
- 8 GPU placement `0-7`；256 train env；global batch 2048；micro batch 32；rollout epoch 4；PPO update epoch 5；actor LR `5e-6`；value LR `1e-4`；clip `0.2`；gamma `.99`；GAE lambda `.95`；
- 官方标准 eval 为 128 fixed seeds；官方文档给出的 `adjust_bottle` 参考数是 π₀.₅ SFT `85.94%`、PPO `96.09%`（此数是外部参考，尚未在当前集群严格重现）。

用户已确定论文所有实验统一 **4 GPU**。资源等效缩放应为：

| 参数 | 官方 8 GPU | 本项目 4 GPU 等效 |
|---|---:|---:|
| component placement | `0-7` | `0-3` |
| train env | 256 | 128 |
| global batch | 2048 | 1024 |
| micro batch | 32 | 32（不缩，保持 per-GPU load） |
| rollout epoch | 4 | 4 |
| fixed-seed eval episode | 128 | 128（不缩） |

“4 GPU 等效配置”不应表述成“官方原始 8-GPU 硬件规模复现”；应表述为算法设置保持官方、资源规模按每卡负载等比缩放。

### 6.1 发现的 4-GPU 评测 OOM

作业 `638607` 的第一训练 rollout 能运行，但评测时 128 个环境并发使 π₀.₅ attention prefix cache OOM（H100 80 GB；报错尝试再分配 458 MiB，GPU 仅剩约 43 MiB）。这不是 Event advantage 逻辑错误。

修复是保持总 128 fixed-seed episode，却分 wave 执行：

```text
env.eval.total_num_envs = 16
env.eval.rollout_epoch = 8
16 × 8 = 128 episodes
PYTORCH_ALLOC_CONF=expandable_segments:True
```

目前这一修复已写入 V2 script；应同样应用到所有 4-GPU GAE/official baseline script，保证公平且不改变 eval 样本总数。

---

## 7. 已实现代码与关键提交

代码根目录：`/home/jj/RLinf-piRL`。

### 7.1 Event 核心实现

核心文件：

- `rlinf/algorithms/event_credit.py`
  - 注册 `event_smdp_interventional`、`event_smdp_temporal`、`event_smdp_residual`。
  - `event_smdp_residual` 实现 V2 的 credit-conserving GAE residual mix；`event_mix_lambda=0` 应严格退化为 GAE。
- `rlinf/algorithms/event_value.py`
  - `OnlineEventValueSidecar`、EMA target value、Event Value 计算、branch endpoint Event Value 推理、sidecar save/load。
- `rlinf/algorithms/event_intervention.py`
  - matched-state branch return 与 influence utility。
- `rlinf/workers/actor/embodied_fsdp_actor_worker.py`
  - sidecar 初始化/恢复、branch supervision counter、actor gate、checkpoint sidecar 保存与 Event metrics。
- `rlinf/workers/rollout/hf/huggingface_worker.py`
  - chunk candidate sampling、branch collection hooks。
- `rlinf/envs/sim/robotwin/robotwin_snapshot.py`、`robotwin_env.py`
  - strict RoboTwin/SAPIEN snapshot 与 branch step 接口。
- `examples/embodiment/event_observer/train_event_observer.py`
  - Observer + Event Value 离线训练；已新增 boundary Precision/Recall/F1、F1@±3 的训练/验证支持。

### 7.2 重要提交（本地 `event-smdp-credit`）

| Commit | 内容 |
|---|---|
| `48751fcf` | V2 residual Event credit、EMA target critic、sidecar resume、chunk gate、Observer boundary P/R/F1；相关单测曾 `13 passed` |
| `ab1627ab` | Event branch collector 的 eval assets 配置 |
| `deda313b` | 官方 RoboTwin π₀.₅ reproduction launchers |
| `81b9e6fe` | branch collector checkpoint cadence 修复 |
| `7020b18a` | 初步 4-GPU 脚本 |
| `f3a62216` | 4-GPU 每卡等效缩放 |
| `375f412e` | eval placement 可设为 `0-3`，默认仍严格 `0-7` |
| `0ceb6489` | 关闭训练/eval video artifact，避免无关视频 I/O 与库兼容问题 |
| `45c2d403` | V2 评测改为 `16 env × 8 rollout` 的 128 episode wave，并加入 expandable allocator；GitHub push 当时网络失败 |
| `f588b8a8` | branch collector 增加 optional sidecar resume，允许跨作业累计 supervision |

### 7.3 可用启动脚本

- `examples/embodiment/slurm_robotwin_pi05_official_sft_eval.sh`
- `examples/embodiment/slurm_robotwin_pi05_official_ppo_repro.sh`
- `examples/embodiment/slurm_robotwin_pi05_action_gae_formal.sh`
- `examples/embodiment/slurm_robotwin_pi05_eventvalue_branch_collect_v2.sh`
- `examples/embodiment/slurm_robotwin_pi05_eventvalue_v2.sh`
- `examples/embodiment/slurm_robotwin_pi05_eventvalue_full.sh`

`slurm_robotwin_pi05_eventvalue_full.sh` 是遗留 V1/硬替换设计（单卡、action-level、signed interventional），**不可用于 V2 主实验**。V2 应只使用 `*_branch_collect_v2.sh` 与 `*_eventvalue_v2.sh`。

### 7.4 仓库状态风险（必须先处理）

本地快照：

```text
branch: event-smdp-credit
HEAD:   f588b8a87d9eadef37c3670721a0ef938eaf2f7a
```

本地存在用户拥有的未提交变更，必须保留，绝不可 reset/checkout 覆盖：

```text
D docs/EVENT_SMDP_INTERVENTIONAL_CREDIT_ZH.md
?? docs/EVENTVALUE_RL_V2_INTEGRATION_REVIEW_ZH.md
?? docs/ICLR_EVENT…任务信用分配设计思路.md
```

HPC 代码树已被其他 CALVIN 工作独立推进：

```text
/data/user/leviccdong/EKSF/code/RLinf-piRL
branch: event-smdp-credit
HEAD:   f61388636474745ea97359f6bc8cb66d71185686
```

HPC 不等于本地分支，且其 CALVIN scripts 有用户/其他工作中的改动。不要 `git reset --hard`，不要盲目强推或整支 merge。本地 `45c2d403` 所改 V2 script 已在 HPC 的后续 commit 中有等价内容；`f588b8a8` 的 branch collector 改动曾用 `scp` 同步到 HPC，因此可能显示为未提交修改。续接时应：

1. `git status --short` 先审计两端；
2. 比较目标文件；
3. 对单个无冲突 commit 做 cherry-pick 或最小补丁；
4. 绝不覆盖 `experiments/calvin_posttrain/*` 的未提交内容。

---

## 8. 模型、环境、数据路径

### 8.1 HPC 基础路径

```text
HPC SSH alias:      hpc
HPC code:           /data/user/leviccdong/EKSF/code/RLinf-piRL
πRL Python env:     /data/user/leviccdong/EKSF/env_pirl_pi05
RoboTwin source:    /data/user/leviccdong/EKSF/runtime/RoboTwin-RLinf_support
RoboTwin assets:    /data/user/leviccdong/EKSF/runtime/etsf_stage0/RoboTwin_v14_dual_gpu_20260831
```

常用启动环境：

```bash
source /share/anaconda3/bin/activate /data/user/leviccdong/EKSF/env_pirl_pi05
export RLINF_LOCAL_RAY=1
unset RAY_ADDRESS
export RLINF_COMPONENT_PLACEMENT=0-3
```

### 8.2 已有模型

```text
# RoboTwin 单任务 SFT
/data/user/leviccdong/EKSF/models/RLinf-Pi05-RoboTwin-SFT-adjust_bottle

# ManiSkill 多任务 SFT
/data/user/leviccdong/EKSF/models/RLinf-Pi05-ManiSkill-25Main-SFT

# π₀.₅ base
/data/user/leviccdong/EKSF/models/pi05_base

# SAM 3.1
/data/user/leviccdong/EKSF/models/sam3.1

# CALVIN ABC→D π₀.₅ SFT（本次会话已完成下载与 SHA-256 两端校验）
/data/user/leviccdong/EKSF/models/RLinf-Pi05-CALVIN-ABC-D-SFT
```

CALVIN 权重完整标记 `.complete` 存在，`model.safetensors` 精确大小为 `7,473,091,464` bytes，并已检查 `metadata.pt`、`InternRobotics/InternData-Calvin_ABC/norm_stats.json`。

### 8.3 Event checkpoints

```text
# Offline Observer
/data/user/leviccdong/EKSF/outputs/robotwin_adjust_bottle_event_pretrain_v1_200plus/best.pt

# RGB student
/data/user/leviccdong/EKSF/outputs/robotwin_adjust_bottle_rgb_student_v1_200plus/best.pt

# 已完成 4-GPU branch debug 的 sidecar
/data/user/leviccdong/EKSF/outputs/robotwin_event_branches_v2_4gpu_equivalent_debug/
  robotwin_adjust_bottle_branch_collect_v2_4gpu_equivalent_debug/
  checkpoints/global_step_2/actor/eventvalue_sidecars.pt
```

读出上述 sidecar 的事实：

```text
branch_supervision_count = 128   # state 数，不是 outcome 数
branch_outcome_count     = 512   # 128 × 4 candidates
```

因此它不足 500-state V2 gate，不能直接作为“已校准 Influence Model”使用。

---

## 9. 实验事实与失败记录

### 9.1 SFT/GAE/V1 Event 历史结果（不可混成正式论文结果）

之前使用单 GPU、`train_env=2`、batch 2/micro 1、`rollout_epoch=1`，并强制 action-level reward/logprob 的小规模 pilot，得到：

| checkpoint | 固定 64 episode success |
|---|---:|
| SFT | 40/64 = 62.5% |
| GAE step 25 | 24/64 = 37.5% |
| GAE step 50 | 32/64 = 50.0% |
| GAE step 75 | 28/64 = 43.75% |
| GAE step 100 | 18/64 = 28.125% |
| V1 EventValue step 25 | 0/64 = 0% |
| V1 EventValue step 92 | 0/64 = 0% |

解释：

- 这些不是官方尺度、不是 chunk-level 官方 πRL 配置，也不是 4-GPU 对照；不能用来断言官方 πRL 无效；
- GAE 从 62.5% 掉到 28.1% 指出 PPO 自身尚未稳定，因此在修复 baseline 前不能解释 Ours；
- V1 0% 是真实失败信号，符合未校准 influence sign 接管 advantage 的机制性诊断。

### 9.2 `638606`、`638607`、`638608`

| Job | 用途 | 状态/结论 |
|---|---|---|
| `638606` | RoboTwin π₀.₅ SFT 官方协议评测 | 完成；本会话未系统抽取/复核其最终指标，不要把它和早先 64-episode 数混淆 |
| `638607` | 4-GPU GAE throughput/stability trial | 训练首轮可运行，但 128 concurrent eval 发生 π₀.₅ eval OOM；Slurm 标记 failed，不能作为 baseline 结果 |
| `638608` | 4-GPU V2 branch debug，冻结 actor | 完成 2 updates；验证 snapshot/branch/sidecar 路径 |

`638608` 的关键日志：

- 每个 global step 512 trajectories；
- rollout `success_once≈0.8398`（step 1）与 `≈0.8242`（step 2）；这是冻结 SFT branch-collection rollout 成功率，**不是 Event PPO 成功率**；
- `event/branch_outcome_count=256` 后至 `512`；最终 sidecar state count 是 128；
- `event/influence_loss≈0.0032--0.0033`，`event/value_smdp_loss` 可反传；
- actor LR=0、critic LR=0、`event/mix_lambda=0`；所以没有 policy update，也没有 Event success 改善的测试；
- 耗时约 22--23 分钟/epoch（128 train env、rollout epoch 4、4 GPU）。

### 9.3 `639000`：V2 branch continuation 的实际结局

原计划：从 `638608` sidecar 恢复，跑 6 epoch，累计约 512 branch state / 2,048 outcomes。

事实：

```text
Job:      639000 robotwin_event_branches_v2
Start:    2026-09-21 20:46:38
End:      2026-09-21 20:50:24
State:    CANCELLED+
ExitCode: 0:0
```

它成功完成 4-GPU/Ray/FSDP/environment 初始化，但在第一个 rollout 前 Slurm 发送 `SIGTERM`，日志为 `JOB 639000 ... CANCELLED`。没有 `global_step=1` metric、没有新 checkpoint、没有新 sidecar。因此：

- 不应说它“跑完了”；
- 不能说 branch state 达到了 512；
- 取消的上层原因未在日志中明确，可能是外部 `scancel`、账户/调度策略或监控操作；不要无证据归咎算法代码。

### 9.4 当前可报告效果结论

| 项目 | 当前结论 |
|---|---|
| Observer 离线拟合 | 有初步 cache 指标，但未证实 online/generalization |
| same-state branch infrastructure | 已在 `638608` 跑通 |
| V1 EventValue-RL | 明确失败（0%） |
| V2 EventValue-RL | 未得到 actor 更新后的 success；正/负增益未知 |
| 官方尺度 πRL baseline | 尚未完成可靠复现；不能与官方 96.09% 进行结论比较 |

---

## 10. 当前 HPC 调度快照与资源约束

本快照时用户 QOS `formal-user` 的上限为：

```text
MaxTRESPerUser: CPU 192, GPU 16, memory 3840G
```

最近 Event pending 的直接原因不是“集群没有 GPU”，而是账户同时 GPU/CPU 配额已被其他作业占满。之后 `639000` 曾开始又被取消。当前快照存在与 Event 独立的 CALVIN/EIPA 作业，例如：

```text
640813  calvin_pirl_pirl_sde_s0_r2     RUNNING  4 GPU
640822  calvin_pirl_pirl_noise_s0_r3   RUNNING  4 GPU
639869  eipa_native_b1_v4              RUNNING  4 GPU
```

这些并非本报告所验证的 Event 结果。续接者在重新提交 Event 前必须先运行：

```bash
ssh hpc 'squeue -u leviccdong -o "%.12i %.28j %.8T %.10M %.5D %R"'
ssh hpc 'sacctmgr -n -P show qos format=Name,MaxTRESPerUser,MaxTRESPU,MaxJobsPU | grep formal-user'
```

若用户希望 Event 优先，不得自行取消 CALVIN/EIPA 作业；必须取得明确授权后再做任何 `scancel`。

---

## 11. 推荐的严格实验序列

### 阶段 A：先恢复稳定的官方算法基线

1. 确认 `638606` 的 128 fixed seed SFT 评测指标、seed manifest、checkpoint 路径；
2. 使用 official chunk-level config（不是早期 action-level pilot）做 4-GPU πRL Flow-SDE + GAE；
3. 使用 `train_env=128, global_batch=1024, micro=32, rollout_epoch=4`；
4. 使用 `eval_env=16, eval_rollout_epoch=8` 保持总 128 eval episode，避免 OOM；
5. 先跑 SFT→step10/25/50/75/100 的 checkpoint 曲线，全部相同 128 evaluator；
6. 若仍出现策略退化，先公平地同时调整 baseline 与 Ours：较低 actor LR、较少 PPO update epoch、较小 clip、KL-to-SFT 或 early-stop、advantage clipping/normalization、critic warm-up。不得只给 Ours 稳定器。

baseline 作为 Event 论文公平对照时，应使用与 Ours 相同的 `chunk_level reward/logprob`；之前“action-level GAE”版本只可作为一个额外消融，不得冒充官方 π₀.₅ 配置。

### 阶段 B：Observer online calibration

在固定 SFT actor 和 early stable πRL actor 上收集 rollout，重新由 simulator oracle 标注；报告：

- relation primitive per-head F1；
- event F1；
- boundary precision/recall/F1 与 F1@±3 steps；
- progress MAE；
- uncertainty calibration（至少 reliability/visibility missing 情况）；
- failure/recovery/occlusion slice。

若 online boundary F1 明显低于 offline，训练数据必须加入 `expert + SFT + early πRL + failure/recovery` rollout，不能直接把 offline cache 93% accuracy 当作可用。

### 阶段 C：Branch dataset 与 Influence pretraining

1. 使用冻结 SFT actor，恢复已有 sidecar：

```text
ROBOTWIN_EVENT_SIDECAR_RESUME=/data/user/leviccdong/EKSF/outputs/robotwin_event_branches_v2_4gpu_equivalent_debug/robotwin_adjust_bottle_branch_collect_v2_4gpu_equivalent_debug/checkpoints/global_step_2/actor/eventvalue_sidecars.pt
```

2. 重新提交 `slurm_robotwin_pi05_eventvalue_branch_collect_v2.sh`，确认它没有被外部取消；
3. 先运行能新增至少 384 state 的 epoch 数；按 `638608` 经验约 64 state/epoch，6 epoch 加现有128约为512 state，但必须以最终 sidecar 的 `branch_supervision_count` 为准；
4. 在 held-out state/episode 上计算 ranking、Kendall-\(\tau\)、sign accuracy；保存分割清单，不能在同一 branch state 上训练和汇报 ranking；
5. 记录 branch transitions，计入 total interaction budget。

示例提交（在资源可用且所有路径存在时）：

```bash
ssh hpc '
  cd /data/user/leviccdong/EKSF/code/RLinf-piRL
  export ROBOTWIN_EVENT_OBSERVER_CKPT=/data/user/leviccdong/EKSF/outputs/robotwin_adjust_bottle_event_pretrain_v1_200plus/best.pt
  export ROBOTWIN_EVENT_RGB_STUDENT_CKPT=/data/user/leviccdong/EKSF/outputs/robotwin_adjust_bottle_rgb_student_v1_200plus/best.pt
  export ROBOTWIN_EVENT_SIDECAR_RESUME=/data/user/leviccdong/EKSF/outputs/robotwin_event_branches_v2_4gpu_equivalent_debug/robotwin_adjust_bottle_branch_collect_v2_4gpu_equivalent_debug/checkpoints/global_step_2/actor/eventvalue_sidecars.pt
  export ROBOTWIN_BRANCH_COLLECT_EPOCHS=6
  export ROBOTWIN_BRANCH_SAVE_INTERVAL=6
  export ROBOTWIN_LOG_PATH=/data/user/leviccdong/EKSF/outputs/robotwin_event_branches_v2_resume512
  export ROBOTWIN_EXPERIMENT_NAME=robotwin_adjust_bottle_branch_collect_v2_resume512
  sbatch --export=ALL examples/embodiment/slurm_robotwin_pi05_eventvalue_branch_collect_v2.sh
'
```

提交后立即检查：

```bash
ssh hpc 'squeue -j <JOBID> -o "%.12i %.28j %.8T %.10M %.5D %R"'
ssh hpc 'tail -f /data/user/leviccdong/EKSF/outputs/robotwin_event_branches_v2_<JOBID>.out'
```

在 checkpoint 完成后不要猜计数，显式读取：

```bash
source /share/anaconda3/bin/activate /data/user/leviccdong/EKSF/env_pirl_pi05
python - <<'PY'
import torch
p = '.../eventvalue_sidecars.pt'
x = torch.load(p, map_location='cpu')
print(x['branch_supervision_count'], x['branch_outcome_count'])
PY
```

### 阶段 D：V2 actor 测试

在 B/C 通过前不要提交正式 Event PPO。通过后从**相同 SFT checkpoint**启用：

```text
algorithm.adv_type=event_smdp_residual
algorithm.reward_type=chunk_level
algorithm.logprob_type=chunk_level
event_credit.granularity=chunk
Event Value / Influence sidecar = C 阶段的 checkpoint
```

推荐 first safe schedule：

```text
min_supervision_for_actor = 500
max_lambda               = 0.10（首轮；稳定后最多 0.25）
warmup_optimizer_steps   = 10
ramp_optimizer_steps     = 50
actor LR                 = 2e-6
PPO update epoch         = 2
clip ratio               = 0.1
critic warm-up           = 10
```

必须记录每个 checkpoint 的 `event/mix_lambda`、branch count、KL、clip fraction、value explained variance、success。若 V2 仍快速低于 SFT，先回退 \(\lambda\)，检查 Influence ranking/sidecar OOD，不应通过隐藏失败 eval 或选择性 checkpoint 得出结论。

### 阶段 E：对照与消融

同一 SFT、seed、interaction budget、eval protocol 下至少：

1. SFT only；
2. πRL Flow-SDE + PPO + GAE；
3. πRL + Observer + GAE（强 auxiliary control：有相同 Observer/labels/额外参数，PPO 仍用普通 GAE）；
4. Event-SMDP uniform（无 intervention）；
5. Event-SMDP temporal/boundary allocation；
6. V2 full residual interventional event credit。

主指标：

- success-vs-env-step AUC；
- 同等 success 所需 total simulator transitions，及 actor-only interaction；
- \(N_{80}/N_{90}\)（若任务成功率范围允许）；
- final fixed-seed success、3 seeds mean/std、稳定性；
- credit ranking accuracy / Kendall-\(\tau\)；
- CALVIN 使用官方 `Avg. Subtasks`, `Len-1` 到 `Len-5`；
- Observer 使用 event/relation F1、boundary F1@tolerance、progress MAE。

branch rollout 与 actor rollout 都必须进 `N_total`：

\[
N_{total}=N_{actor-rollout}+N_{branch-rollout}.
\]

---

## 11A. 新增：分支标签信号审计与 oracle 诊断阶梯

这一节是 V2 进入任何 actor 更新前的**硬性诊断门**。它回答的不是“Influence loss 小不小”，而是三个可区分的问题：

1. 同一状态的 Flow-SDE action candidates 是否真的造成不同的未来后果；
2. 若有差异，当前 `I_\xi` 是否能预测候选排序，而不是输出塌缩的常数；
3. 若最终 PPO 没有提升，究竟是 Event Value/时间抽象、Observer，还是 Influence 的问题。

### 11A.1 128 个旧 state 的可用信息与不可用信息

旧的 `638608`/V2 debug sidecar 保存了：

```text
branch_supervision_count = 128 states
branch_outcome_count     = 512 candidates (= 128 × 4)
```

但它**没有持久化每个 state 的四个 `Y(s,a_k)`、候选 action 或四个预测 score**。因此历史上的 `influence_loss≈0.0032` 不能被解释为“Influence 学会了”：若四个 branch return 本就几乎相等，预测全零也会有很小 MSE；也无法从该 sidecar 反算 ranking、Kendall-\(\tau_b\) 或 top-1 regret。不得把这 128 state 作为已通过 signal gate 的证据。

已新增持久化审计路径。每一个新鲜 branch batch 会按 FSDP rank / PPO update 写出紧凑的：

```text
branch_returns        [num_states, K]
predicted_scores      [num_states, K]
event_ids             [num_states]
branch_success        [num_states, K]   # 短 branch 内环境报告的 success/termination
success_available     # 该环境是否提供 success 字段
```

文件仅保存标量诊断，不保存 RGB。分析器为：

```text
examples/embodiment/event_observer/analyze_branch_diagnostics.py
```

它已通过 synthetic NPZ 测试；代码的静态编译与 Slurm shell syntax 也通过。真实 GPU rollout 仍必须单独验证 action-transport 与 simulator branch 路径。

### 11A.2 必须执行的新鲜 held-out branch audit

必须从冻结的 `adjust_bottle` π₀.₅ SFT actor 采集**独立于 Influence 训练的** 128 state 审计集。建议复用当前 sidecar 只作为被评估的 `I_\xi`，并设定：

```text
actor learning rate = 0
Event Value learning rate = 0
Influence record_only = true            # 不做 I_ξ optimizer step，也不计入训练 supervision
K = 4
H = 当前 branch horizon（先固定 10）
```

这份采集消耗的 branch transitions 必须单列、并纳入未来总 interaction budget；它不是训练数据。若需要之后训练 `I_\xi`，训练集与该 held-out audit 的 simulator state 必须按 episode/state seed 分开，不能同一状态既训练又报 ranking。

对每个 state 定义：

\[
Y_k=R_{t:t+H}^{(k)}+\gamma^H\bar V_E(z_{t+H}^{E,(k)}),\qquad k\in\{1,\ldots,K\}.
\]

审计报告必须包含下表所有项目：

| 统计 | 计算与解释 |
|---|---|
| 每 state `std(Y_1,...,Y_K)` 和 range | 分支候选是否导致可区分后果；同时报均值、中位数、分位数与 near-zero fraction。 |
| 零预测器 MSE | 对中心化 candidate-0 target `Y_0-mean_k(Y_k)` 的全零 MSE；必须与模型 MSE 和 MSE 比率一起报告。 |
| 按 event 的 return spread | 找出真正有决策差异的事件；不能只报总体平均。 |
| tie-aware pairwise accuracy | 忽略 `|Y_i-Y_j|≤ε` 的近似平局，避免把数值噪声当排序成功。 |
| Kendall-\(\tau_b\) | 对每个 state 的候选排序做 tie-corrected 汇总。 |
| top-1 regret | `max_k Y_k-Y_{argmax \hat Y_k}`；也报 exact-best fraction。 |
| predicted-score std | 预测候选 score 的每-state 标准差与 near-constant fraction，直接检测 `I_\xi` 输出塌缩。 |
| success/failure 分层 | 分别报 candidate-0 success/failure、任一 candidate success/无 candidate success 的上述指标；这里的 success 是短 branch 环境返回，不应误称作完整主 episode success。 |

默认 `ε=1e-4`，但应对 reward/event-value 的量纲做敏感性分析，而不能借一个过大的 `ε` 隐藏失败排序。

如果大部分 state 的 spread 接近零，问题不应先归咎于网络，更可能是：候选 action samples 太相似、branch horizon 太短、采样位置不在关键决策点，或 SFT 成功率约 84% 接近该任务 ceiling。此时继续同分布采 500 state 没有科学价值；应先增大 candidate diversity（可控温度/噪声但保持 flow protocol）、延长 `H`、以 boundary/不确定性/失败恢复状态做采样，或换更有决策分歧的任务。

### 11A.3 oracle 诊断阶梯：从哪里坏，必须可定位

不要直接从 GAE 跳到 full V2。所有 rung 必须保持同一 SFT 初始化、Flow-SDE/PPO 其余超参、seed、主 rollout 预算与**主 rollout + branch rollout**总预算；只改变下表指定变量。

| rung | credit 来源 | 要回答的问题 |
|---:|---|---|
| 1 | 原始 action/chunk-level GAE baseline | πRL/PPO 本身是否稳定。 |
| 2 | **oracle boundary/event** + Event-SMDP uniform | 去掉 intervention 后，Event Value、事件时间尺度、守恒 credit 公式是否有独立价值。 |
| 3 | oracle event + **真实 branch ranking** | 在完美事件条件下，受控干预本身是否有潜力。 |
| 4 | predicted event + 真实 branch ranking | 对比 3，隔离 Observer 的在线误差。 |
| 5 | oracle event + learned `I_\xi` | 对比 3，隔离 Influence 估计误差。 |
| 6 | predicted event + learned `I_\xi` | 完整可部署方法；比较模块误差叠加与 policy drift。 |

预注册式解释规则：

- rung 2 不优于 rung 1：优先检查 Event Value target、事件持续时间 \(D_j\)、SMDP discount 与 residual/uniform credit 公式；此时不能把失败归咎于 `I_\xi`。
- rung 3 优于 rung 2：same-state intervention 对 credit 有可用信息；这是对核心机制最直接的证据。
- rung 4 相对 rung 3 明显下降：Observer（online boundary/role/progress）是瓶颈，先报/修在线 Boundary F1、relation F1、progress MAE。
- rung 5 相对 rung 3 明显下降：Influence Model 是瓶颈，优先看 11A.2 的 spread/ranking，而不是继续更新 actor。
- rung 6 额外下降：两个模块误差叠加，或 PPO 后 policy drift 造成 Observer/`I_\xi` OOD；需要混入 early-policy/failure rollouts 重校准，或降低 \(\lambda\)。

“真实 branch ranking”必须有粒度纪律：当前反事实采的是 action **chunk**，所以 rung 3/4 的 oracle ranking 只能对已采样的 chunk 直接赋 credit。若要把它分给 event 内其他 chunk，必须显式说明那部分仍是插值，或为那些 chunk 也采真实 branch；不能把一个 chunk 的 counterfactual 标签伪装为每个低层 action 的 oracle。

### 11A.4 两个强 falsification control

除上述六项外，完整方法（至少 rung 6）必须有两个同预算、同一 branch state 集的反证对照：

1. **shuffle influence**：在保持事件内 score 分布/尺度的条件下，随机置换 valid influence 到其他 state/chunk；
2. **reverse influence**：反转同一事件内 candidate 或 chunk 的 influence 排序（等价于把高 influence 当低 influence）。

真实 influence 必须同时优于 shuffle 和 reverse，且 branch audit 的 tie-aware ranking 明显高于随机水平，才可以声称“interventional causal credit 生效”。如果真实 influence 与 shuffle 无差别，结论只能是“该配置下额外 intervention 计算没有带来可测收益”，不能以 GNN 或 event 名称替代因果证据。

### 11A.5 RoboTwin 的 oracle 实现前置条件

当前 RoboTwin `adjust_bottle` 在线环境尚未把 task-level event oracle（event id、boundary、progress）以独立字段完整输送到 `EnvOutput`。因此 rung 2--5 不是改一个 config 就能严谨完成：必须先从 simulator task/articulation 状态定义并验证 `adjust_bottle` 的 oracle predicates，写入 rollout label，但**绝不**输入给在线 Observer。不能把预测 event 冒充 oracle event。

ManiSkill 的 event-oracle 路径可能更容易先做机制 ladder，但若 RoboTwin 是主实验，必须补 RoboTwin adapter。Oracle 仅用于诊断、监督和评估；部署版本仍使用 RGB/proprio Observer。

## 12. CALVIN 相关状态与下一步

### 12.1 已下载 checkpoint

本会话已经下载并传输：

```text
RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT
→ /data/user/leviccdong/EKSF/models/RLinf-Pi05-CALVIN-ABC-D-SFT
```

下载过程：HPC 直连 Hugging Face 与 `hf-mirror.com` 均连接超时；改为本机 `HF_HUB_DISABLE_XET=1` 可续传下载，再用 `rsync` 上传 HPC。两端 SHA-256 一致。未来若需下载大模型，优先复用这个流程；不要把只有 `.incomplete` 的 staging directory 当 checkpoint。

### 12.2 当前 CALVIN 作业

HPC 有独立 CALVIN πRL SDE/Noise 作业在运行（job id 随重试变化，快照时为 `640813`, `640822`）。这些作业由 HPC 后续 CALVIN code/monitor changes 支持，**本报告没有验证它们完成、有效、符合 ABC→D 防泄漏或可与 Event 比较**。在论文结果中使用之前必须逐项核验：

- SFT 路径是否为 ABC→D checkpoint；
- online training data/environment 是否仅 A/B/C；
- D 是否 zero-update evaluation；
- Flow-SDE/Flow-Noise、seed、interaction budget、eval manifest；
- 失败重试是否保留完整 provenance，不能只汇报成功 seed。

---

## 13. 常见误表述与禁止事项

1. **禁止**说“YOLOE/GNN 图提供因果证据”。Observer 是表征；same-state different-action branch 才是因果干预证据。
2. **禁止**用 oracle mask/depth/object pose 做主输入后再声称纯 RGB/真机可迁移；oracle 只能作 labels/eval。
3. **禁止**线上查 SAM Zarr cache；online 新状态不在 cache 中。cache 只训练 teacher/student/Observer。
4. **禁止**把 `boundary accuracy=0.9305` 当作 boundary quality；必须给 F1/recall/tolerance。
5. **禁止**把冻结 actor 的 branch collection `success_once≈84%` 说成 Event 提升；它只是 SFT policy rollout。
6. **禁止**把 V1 0% 说成 Event 方法最终效果；它是已定位的 V1 失稳设计失败。
7. **禁止**把早期 1-GPU action-level pilot 说成官方 πRL reproduction；它不是官方 chunk-level/8GPU profile。
8. **禁止**忽略 branch interaction budget；否则 sample-efficiency claim 不公平。
9. **禁止**在 CALVIN D 上训练 Observer/Value/Influence/PPO 后仍称 ABC→D。
10. **禁止**重置或覆盖当前 dirty worktree，尤其 HPC 的 `experiments/calvin_posttrain/*`。

---

## 14. 新接手者的最短行动清单

1. 读本文件，运行 `git status --short` 本地与 HPC；确认没有覆盖用户变更。
2. `squeue -u leviccdong` 检查 16-GPU/192-CPU QOS；确认有 4 GPU + 48 CPU 配额才重新投 Event branch collection。
3. 先恢复并完成 V2 branch continuation，读取 `branch_supervision_count`；若被取消，查 Slurm cancel 来源/账户监控，不要假装有数据。
4. 完成 held-out Influence ranking/sign validation；不足则继续冻结 SFT branch collect。
5. 同时修复并运行 4-GPU chunk-level πRL GAE baseline，使用 16×8 eval；拿到稳定 SFT→PPO 曲线后再比较 Event。
6. 只在 branch/Observer/baseline gates 都通过后，提交 V2 residual PPO；从 \(\lambda\le0.1\) 开始。
7. 所有结果记录任务、SFT、git SHA、config、seed manifest、actor/branch interaction、eval protocol、Slurm job id 与原始日志路径。
8. RoboTwin 得到机制结果后，再按 benchmark-internal data protocol 做 CALVIN ABC→D；RobotWin 多 archetype Observer 扩展作为附录/表征证据，不拿它污染 CALVIN 主对比。

---

## 15. 有用的命令

### 作业状态与日志

```bash
ssh hpc 'squeue -u leviccdong -o "%.12i %.28j %.8T %.10M %.5D %R"'
ssh hpc 'sacct -j <JOBID> --format=JobID,JobName%28,State,ExitCode,Elapsed,Start,End -X'
ssh hpc 'tail -120 /data/user/leviccdong/EKSF/outputs/<PREFIX>_<JOBID>.out'
ssh hpc 'tail -120 /data/user/leviccdong/EKSF/outputs/<PREFIX>_<JOBID>.err'
```

### V2 sidecar 存在性与计数

```bash
ssh hpc '
  source /share/anaconda3/bin/activate /data/user/leviccdong/EKSF/env_pirl_pi05
  python - <<"PY"
import torch
p = "/data/user/leviccdong/EKSF/outputs/.../eventvalue_sidecars.pt"
x = torch.load(p, map_location="cpu")
print("states", x["branch_supervision_count"])
print("outcomes", x["branch_outcome_count"])
PY
'
```

### 代码最小验证

```bash
cd /home/jj/RLinf-piRL
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  test_event_credit.py test_event_value_sidecar.py test_event_intervention.py
bash -n examples/embodiment/slurm_robotwin_pi05_eventvalue_v2.sh
bash -n examples/embodiment/slurm_robotwin_pi05_eventvalue_branch_collect_v2.sh
```

测试文件路径若随分支迁移，应先用 `rg --files | rg 'test_event_(credit|value_sidecar|intervention)'` 定位，不应假定当前工作目录。

---

## 16. 最终交接结论

项目已从“YOLOE/GNN 阶段识别 + CfC”转向“Role-Graph Event Observer + Event Value + same-state intervention credit”的清晰路线；CfC 已从主路径删除，`causal tracker` 也不应再作为方法名。工程上，V2 residual estimator、EMA Event Value、branch snapshot/rollout、sidecar checkpoint、4-GPU scaling 与 CALVIN π₀.₅ checkpoint 已经具备；科研上，当前只证明了离线 Observer 可拟合与 branch 链路可运行，**没有证明 EventValue-RL 提升 πRL**。最重要的后续工作不是扩网络或扩平台，而是：先用足量、独立验证的 branch supervision 校准 (I_\xi)，先稳定官方尺度 chunk-level πRL baseline，再以守恒 residual credit 做同预算、3-seed 的可证伪比较。
