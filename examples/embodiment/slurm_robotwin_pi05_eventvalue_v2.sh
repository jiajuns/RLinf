#!/usr/bin/env bash
# Conservative EventValue-RL V2: calibrated chunk intervention is a residual
# correction to the identical stabilised πRL/GAE control, never a hard GAE replacement.
#SBATCH --job-name=robotwin_eventvalue_v2
#SBATCH --partition=acd_ue
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/robotwin_eventvalue_v2_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/robotwin_eventvalue_v2_%j.err

set -euo pipefail
export RLINF_LOCAL_RAY=1
unset RAY_ADDRESS
export RLINF_RAY_TMPDIR="/tmp/rlinf_ray_${SLURM_JOB_ID:?}"
export RLINF_COMPONENT_PLACEMENT="${RLINF_COMPONENT_PLACEMENT:-0-3}"
export REPO_PATH="${REPO_PATH:-/data/user/leviccdong/EKSF/code/RLinf-piRL}"
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/data/user/leviccdong/EKSF/runtime/RoboTwin-RLinf_support}"
export ROBOTWIN_ASSETS_PATH="${ROBOTWIN_ASSETS_PATH:-/data/user/leviccdong/EKSF/runtime/etsf_stage0/RoboTwin_v14_dual_gpu_20260831}"
export ROBOTWIN_PI05_MODEL="${ROBOTWIN_PI05_MODEL:-/data/user/leviccdong/EKSF/models/RLinf-Pi05-RoboTwin-SFT-adjust_bottle}"
export ROBOTWIN_EVENT_OBSERVER_CKPT="${ROBOTWIN_EVENT_OBSERVER_CKPT:?set Event Observer best.pt}"
export ROBOTWIN_EVENT_RGB_STUDENT_CKPT="${ROBOTWIN_EVENT_RGB_STUDENT_CKPT:?set RGB student best.pt}"
export ROBOTWIN_EVENT_SIDECAR_RESUME="${ROBOTWIN_EVENT_SIDECAR_RESUME:?set branch-collection eventvalue_sidecars.pt}"
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export ROBOTWIN_MAX_EPOCHS="${ROBOTWIN_MAX_EPOCHS:-1000}"
export ROBOTWIN_TRAIN_ENVS="${ROBOTWIN_TRAIN_ENVS:-128}"
export ROBOTWIN_EVAL_ENVS="${ROBOTWIN_EVAL_ENVS:-128}"
export ROBOTWIN_GLOBAL_BATCH="${ROBOTWIN_GLOBAL_BATCH:-1024}"
export ROBOTWIN_MICRO_BATCH="${ROBOTWIN_MICRO_BATCH:-32}"
export ROBOTWIN_LOG_PATH="${ROBOTWIN_LOG_PATH:-/data/user/leviccdong/EKSF/outputs/robotwin_eventvalue_v2}"
export ROBOTWIN_EXPERIMENT_NAME="${ROBOTWIN_EXPERIMENT_NAME:-robotwin_adjust_bottle_eventvalue_v2}"

cd "${ROBOTWIN_PATH}"
source /share/anaconda3/bin/activate /data/user/leviccdong/EKSF/env_pirl_pi05
exec python "${REPO_PATH}/examples/embodiment/train_embodied_agent.py" \
  --config-path config --config-name robotwin_adjust_bottle_ppo_openpi_pi05 \
  runner.max_epochs="${ROBOTWIN_MAX_EPOCHS}" runner.logger.log_path="${ROBOTWIN_LOG_PATH}" \
  runner.logger.experiment_name="${ROBOTWIN_EXPERIMENT_NAME}" runner.val_check_interval=25 runner.save_interval=25 \
  env.train.total_num_envs="${ROBOTWIN_TRAIN_ENVS}" env.eval.total_num_envs="${ROBOTWIN_EVAL_ENVS}" \
  env.train.rollout_epoch=4 env.eval.rollout_epoch=1 env.train.assets_path="${ROBOTWIN_ASSETS_PATH}" env.eval.assets_path="${ROBOTWIN_ASSETS_PATH}" \
  actor.model.model_path="${ROBOTWIN_PI05_MODEL}" rollout.model.model_path="${ROBOTWIN_PI05_MODEL}" \
  +rollout.unnorm_key=adjust_bottle +rollout.collect_transitions=true \
  algorithm.reward_type=chunk_level algorithm.logprob_type=chunk_level algorithm.adv_type=event_smdp_residual algorithm.loss_type=actor_critic \
  algorithm.update_epoch=2 algorithm.clip_ratio_high=0.1 algorithm.clip_ratio_low=0.1 \
  +algorithm.event_value_source=learned_sidecar +algorithm.event_boundary_threshold=0.5 \
  +algorithm.event_sidecar.observer_checkpoint="${ROBOTWIN_EVENT_OBSERVER_CKPT}" +algorithm.event_sidecar.rgb_student_checkpoint="${ROBOTWIN_EVENT_RGB_STUDENT_CKPT}" \
  +algorithm.event_sidecar.resume_sidecar_path="${ROBOTWIN_EVENT_SIDECAR_RESUME}" +algorithm.event_sidecar.value_lr=1.0e-4 +algorithm.event_sidecar.target_ema_decay=0.995 +algorithm.event_sidecar.proprio_time_delta=5.0 \
  +algorithm.event_branch.num_candidates=4 +algorithm.event_branch.chunk_interval=10 +algorithm.event_branch.horizon=10 +algorithm.event_branch.min_supervision=500 +algorithm.event_branch.influence_lr=1.0e-4 \
  +algorithm.event_credit.granularity=chunk +algorithm.event_credit.min_supervision_for_actor=500 +algorithm.event_credit.max_lambda=0.25 +algorithm.event_credit.warmup_optimizer_steps=10 +algorithm.event_credit.ramp_optimizer_steps=50 +algorithm.event_credit.influence_beta=0.02 +algorithm.event_credit.influence_clip=3.0 \
  actor.optim.lr=2.0e-6 actor.optim.critic_warmup_steps=10 actor.model.openpi.noise_method=flow_sde +actor.model.openpi.joint_logprob=false actor.model.openpi.value_after_vlm=false \
  actor.global_batch_size="${ROBOTWIN_GLOBAL_BATCH}" actor.micro_batch_size="${ROBOTWIN_MICRO_BATCH}"
