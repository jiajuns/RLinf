# Event-SMDP Interventional Credit：面向 Flow-VLA 的长程信用分配

## 一句话概述

本方法以 **π0.5 + πRL Flow-SDE/PPO** 为唯一策略与强化学习底座，不重写 VLA 或 PPO；我们在其上引入事件时间抽象和同状态动作干预。Event Observer 将连续操作转为带持续时间的语义事件，Event Value Critic 估计事件对任务成功的长期价值，再从完全相同的 simulator state 对多个 Flow-SDE action branch 做短 rollout，测量哪个动作真正改变未来事件价值。该干预影响用于把一个事件的 SMDP advantage 分配给事件内部关键动作，从而替代 GAE 按时间距离传播信用的方式。

核心论断是：

\[
\boxed{\text{事件价值提供长程时间抽象；同状态动作干预提供因果 credit evidence；二者使 Flow-VLA 获得更准确且更省样本的信用分配。}}
\]

## 完整目标架构

```mermaid
flowchart LR
    A[RGB observation] --> B[冻结 YOLOE]
    B --> C[object / target / gripper\nboxes, confidence, ROI]
    C --> D[Role-Graph Event Observer\n时序角色图 + 2-layer GNN]
    D --> E[zE: event state\nq(e): event posterior\np: progress / b: boundary / u: uncertainty]

    A --> P[π0.5 Flow-SDE policy]
    P --> Q[πRL rollout action]
    Q --> S[ManiSkill simulator]
    S --> A

    E --> V[Event Value Critic VE(zE)]
    S --> T[trajectory: r, done, action, event]
    V --> U[Event-SMDP target]
    T --> U

    S --> X[保存关键 state snapshot]
    P --> Y[采样 K≈4 Flow-SDE actions]
    X --> Z[每个 action short rollout H=10~20]
    Y --> Z
    Z --> I[Interventional influence I(s,a)]
    V --> Z

    U --> W[事件 advantage AE]
    I --> W
    W --> PPO[原始 πRL PPO actor update]
    PPO --> P
```

架构分为四个职责明确的模块：

1. **Role-Graph Event Observer**：从视觉关系构成事件状态，不把检测器输出错误地当作关系或因果真值。
2. **Event Value Critic**：在事件粒度传播长程回报，而非只预测帧级 phase。
3. **Controlled Action Intervention**：固定同一 simulator state，仅改变 action，形成真正的反事实对照。
4. **πRL PPO 更新**：策略分布、Flow-SDE 采样与 PPO ratio 均沿用官方 πRL，只替换 advantage 的构造与分配。

## 模块一：Role-Graph Event Observer

### 输入与图构建

YOLOE 冻结，仅从 RGB 产生 object、target、gripper 的候选框、置信度与 ROI feature。角色图的节点包含：

- 角色类别与检测置信度；
- bounding box、ROI feature、轨迹与缺失标记；
- 歧义、遮挡及检测不确定性特征。

边包含 object-target、gripper-object、gripper-target 等关系的相对位置、尺寸比、IoU、距离和时序变化。两层 GNN 在帧内传播关系，再以短历史聚合形成事件状态：

\[
z_t^E = f_{\text{event}}(G_{t-L:t},\;\phi_{\text{global}}(I_t),\;\phi_{\text{ROI}}(I_t)).
\]

### 输出

Observer 的主输出是：

- 事件 posterior \(q_t(e)\)：例如 approach、grasp、lift、transport、align、place；
- 事件状态 \(z_t^E\)；
- 事件进度 \(p_t\in[0,1]\)；
- 边界概率 \(b_t\)；
- 不确定性 \(u_t\)。

抓持、承托、局部关系、目标区域等可作为辅助监督，但不再是论文的最终因果结论。GNN 只是 observation model；因果证据必须来自同状态、不同 action 的受控干预。

## 模块二：Event-SMDP Value Critic

将同一事件 ID 的连续时间步构成事件段 \(j\)，其起点、终点与时长为 \((\tau_j,\tau_{j+1},D_j)\)。Event Critic 估计：

\[
V_E(z_{\tau_j}^E) \approx \mathbb{E}[\text{从当前事件状态到任务终止的折扣回报}].
\]

事件的半马尔可夫回报与 TD residual 为：

\[
R_j=\sum_{k=0}^{D_j-1}\gamma^k r_{\tau_j+k},\qquad
\delta_j^E=R_j+\gamma^{D_j}V_E(z_{\tau_{j+1}}^E)-V_E(z_{\tau_j}^E).
\]

终止事件不 bootstrap。该目标让 critic 在“完成 grasp 后仍有多少成功价值”这一更稳定的时间尺度学习，而不是让帧级 critic 跨越数十个相互异质的动作传播回报。

## 模块三：同状态 Flow-SDE action intervention

在高 boundary 概率、低 event value 或关键事件状态保存完整 simulator snapshot。令策略从同一状态 \(s_t\) 采样 \(K\approx4\) 个 Flow-SDE candidate action：

\[
a_t^{(1)},\ldots,a_t^{(K)}\sim\pi_{\theta}^{\mathrm{Flow\text{-}SDE}}(\cdot\mid s_t).
\]

每个 action 均从同一快照恢复并执行 \(H=10\sim20\) 步短 rollout：

\[
Y(s_t,a_t^{(k)})=\sum_{h=0}^{H-1}\gamma^h r_{t+h}^{(k)}+
\gamma^H V_E(z_{t+H}^{E,(k)}).
\]

干预影响定义为候选分支的中心化未来事件价值：

\[
I(s_t,a_t^{(k)})=Y(s_t,a_t^{(k)})-
\frac1K\sum_{k'=1}^{K}Y(s_t,a_t^{(k')}).
\]

这不是用观察相关性推测“哪个动作重要”，而是控制初始状态不变、只改变 action 后得到的因果差异。

## 模块四：事件内部 credit 分配与 PPO

事件 \(j\) 获得 \(\delta_j^E\) 后，不再向该段每个 timestep 平均复制，也不使用 GAE 的指数时间传播。对事件内动作归一化干预影响：

\[
w_t=\operatorname{softmax}_{t\in j}\left(I(s_t,a_t)/T_I\right),
\qquad
A_t^{\mathrm{IEVC}}=D_j\,w_t\,\delta_j^E.
\]

其中 \(D_j\) 保证事件内 action advantage 的平均尺度保持为事件 advantage，\(T_I\) 是 influence temperature。随后将 \(A_t^{\mathrm{IEVC}}\) 直接输入原始 πRL PPO actor loss：

\[
\mathcal{L}_{\mathrm{PPO}}=-\mathbb{E}_t\left[
\min\left(r_t A_t^{\mathrm{IEVC}},
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t^{\mathrm{IEVC}}\right)\right].
\]

因此 VLA、Flow-SDE sampler、PPO ratio 与策略更新机制均未被替换；创新严格集中在 credit estimator。

## 创新点表述

### 核心创新

传统 GAE 对长程机器人任务的每个 timestep 按时间距离传播回报，无法区分“真正改变后续抓取/放置前景的动作”和仅发生在同一事件中的动作。我们提出 Event-SMDP Interventional Credit：用事件状态与事件价值把长程任务压缩到语义稳定的 SMDP 时间尺度，再从完全相同的 simulator state 出发，通过多个 Flow-SDE action branch 的受控 rollout 测量动作对未来事件价值的真实变化，并将事件 advantage 依据该干预影响分配给关键动作。方法同时解决长程 value propagation 与细粒度 causal credit assignment，且不需要重写 πRL 或 VLA。

### 与常见替代方案的区别

| 方法 | 长程时间抽象 | 关键 action 依据 | 因果证据 |
|---|---|---|---|
| GAE | 无，逐 timestep | 时间距离 | 无 |
| phase 分类辅助任务 | 有标签但不进入 value | 分类置信度 | 无 |
| Temporal Event Credit | 事件段 | 事件内均匀/启发式权重 | 无 |
| 本方法 | Event-SMDP value | future event value 差异 | 同 state、不同 action branch |

## 实验设计与指标

### 分阶段验证

1. **官方 πRL 复现**：π0.5 + πRL Flow-SDE/PPO，固定任务、seed、online interaction budget。
2. **ManiSkill Oracle Event-SMDP**：先以 simulator privileged event 做因果 credit 机制验证，排除 learned observer 误差。
3. **Intervention ablation**：比较 GAE、Event-SMDP temporal credit、启发式 temporal event credit、interventional event-value credit。
4. **Learned Event Observer**：以 YOLOE + Role-Graph GNN 替换 oracle event，报告感知误差对 RL 的影响。
5. **跨基准扩展**：CALVIN、LIBERO-Long/LIBERO-PRO，重点考察同等 success 下的在线交互节省。

### 主指标

- 成功率随 environment steps 的曲线与 AUC；
- 达到相同 success 所需交互数 \(N_{80},N_{90}\)，目标是减少 30--50%；
- 长链任务完成长度；
- Event Critic explained variance；
- credit ranking accuracy：高 influence action 是否更能改变后续 event value/成功；
- 多随机种子均值、标准差与置信区间。

### Observer 指标

- event classification F1；
- boundary F1；
- progress MAE；
- uncertainty calibration；
- Oracle Event 与 Learned Event 的 RL performance gap。

## 当前代码状态与边界

当前 `event-smdp-credit` 分支已经跑通的是 **Oracle Event-SMDP temporal control**：

- ManiSkill 每个 executed action 产生 approach/grasp/lift/transport/align/place 的 oracle `event_id`、`progress` 与 `boundary`；
- 标签经过 `EnvOutput → ChunkStepResult → Trajectory → rollout_batch` 进入 PPO；
- `event_smdp_temporal` 使用上述 SMDP target；
- 暂以 π0.5 原 value head 作为 chunk-level Event Value proxy，并在 action 维度扩展；
- 事件内 influence 暂为零，因此该阶段是均匀 temporal allocation，对应重要的 oracle 对照，而非完整创新结果。

代码中已经有 simulator snapshot/restore 及 branch return/influence utility；但 **learned Role-Graph Event Observer、独立 Event Value Critic、K 分支 rollout 的在线采样与 influence 写回 batch 尚待接入训练循环**。在这些模块接通前，任何实验结论只能称为 Oracle Event-SMDP temporal credit，不应称为完整的 interventional Event Value 方法。

## 代码映射

| 功能 | 位置 |
|---|---|
| Oracle event 标注 | `rlinf/envs/sim/maniskill/maniskill_env.py` |
| Event trajectory transport | `rlinf/data/schema/embodied_types.py`、`embodied_trajectory_builder.py`、`rlinf/workers/env/env_worker.py` |
| Event-SMDP estimator | `rlinf/algorithms/event_credit.py` |
| Same-state branch utility | `rlinf/algorithms/event_intervention.py` |
| Snapshot/restore | `rlinf/envs/sim/maniskill/maniskill_offload_env.py` |
| πRL actor integration | `rlinf/workers/actor/embodied_fsdp_actor_worker.py` |
| Oracle Event-SMDP config | `examples/embodiment/config/maniskill_ppo_openpi_pi05_event_smdp.yaml` |
