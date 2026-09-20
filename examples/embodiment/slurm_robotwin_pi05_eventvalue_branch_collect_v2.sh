#!/usr/bin/env bash
# Freeze π0.5 SFT and collect matched-state branch labels before Event PPO.
# This job intentionally performs no actor/π0.5 critic learning: it only
# updates V_E, I_xi and its EMA target.  Its branch interactions must be added
# to the final method's total simulator-interaction budget.
#SBATCH --job-name=robotwin_event_branches_v2
#SBATCH --partition=acd_ue
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/robotwin_event_branches_v2_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/robotwin_event_branches_v2_%j.err

set -euo pipefail
export RLINF_LOCAL_RAY=1
unset RAY_ADDRESS
export RLINF_RAY_TMPDIR="/tmp/rlinf_ray_${SLURM_JOB_ID:?}"
export RLINF_COMPONENT_PLACEMENT="${RLINF_COMPONENT_PLACEMENT:-0}"
export REPO_PATH="${REPO_PATH:-/data/user/leviccdong/EKSF/code/RLinf-piRL}"
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/data/user/leviccdong/EKSF/runtime/RoboTwin-RLinf_support}"
export ROBOTWIN_ASSETS_PATH="${ROBOTWIN_ASSETS_PATH:-/data/user/leviccdong/EKSF/runtime/etsf_stage0/RoboTwin_v14_dual_gpu_20260831}"
export ROBOTWIN_PI05_MODEL="${ROBOTWIN_PI05_MODEL:-/data/user/leviccdong/EKSF/models/RLinf-Pi05-RoboTwin-SFT-adjust_bottle}"
export ROBOTWIN_EVENT_OBSERVER_CKPT="${ROBOTWIN_EVENT_OBSERVER_CKPT:?set Event Observer best.pt}"
export ROBOTWIN_EVENT_RGB_STUDENT_CKPT="${ROBOTWIN_EVENT_RGB_STUDENT_CKPT:?set RGB student best.pt}"
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

export ROBOTWIN_BRANCH_COLLECT_EPOCHS="${ROBOTWIN_BRANCH_COLLECT_EPOCHS:-650}"
export ROBOTWIN_TRAIN_ENVS="${ROBOTWIN_TRAIN_ENVS:-2}"
export ROBOTWIN_GLOBAL_BATCH="${ROBOTWIN_GLOBAL_BATCH:-2}"
export ROBOTWIN_MICRO_BATCH="${ROBOTWIN_MICRO_BATCH:-1}"
export ROBOTWIN_LOG_PATH="${ROBOTWIN_LOG_PATH:-/data/user/leviccdong/EKSF/outputs/robotwin_event_branches_v2}"
export ROBOTWIN_EXPERIMENT_NAME="${ROBOTWIN_EXPERIMENT_NAME:-robotwin_adjust_bottle_branch_collect_v2}"

cd "${ROBOTWIN_PATH}"
source /share/anaconda3/bin/activate /data/user/leviccdong/EKSF/env_pirl_pi05

# RLinf requires save_interval to be divisible by val_check_interval.  The
# short collector has no validation before epoch 100, but keeps a valid
# checkpoint cadence for the formal 500+ state collection.
exec python "${REPO_PATH}/examples/embodiment/train_embodied_agent.py" \
  --config-path config --config-name robotwin_adjust_bottle_ppo_openpi_pi05 \
  runner.max_epochs="${ROBOTWIN_BRANCH_COLLECT_EPOCHS}" \
  runner.logger.log_path="${ROBOTWIN_LOG_PATH}" runner.logger.experiment_name="${ROBOTWIN_EXPERIMENT_NAME}" \
  runner.val_check_interval=100 runner.save_interval=100 \
  env.train.total_num_envs="${ROBOTWIN_TRAIN_ENVS}" env.train.rollout_epoch=1 env.eval.total_num_envs=1 env.eval.rollout_epoch=1 \
  env.train.assets_path="${ROBOTWIN_ASSETS_PATH}" env.eval.assets_path="${ROBOTWIN_ASSETS_PATH}" \
  actor.model.model_path="${ROBOTWIN_PI05_MODEL}" rollout.model.model_path="${ROBOTWIN_PI05_MODEL}" \
  +rollout.unnorm_key=adjust_bottle +rollout.collect_transitions=true \
  algorithm.reward_type=chunk_level algorithm.logprob_type=chunk_level \
  algorithm.adv_type=event_smdp_residual algorithm.loss_type=actor_critic \
  +algorithm.event_value_source=learned_sidecar +algorithm.event_boundary_threshold=0.5 \
  +algorithm.event_sidecar.observer_checkpoint="${ROBOTWIN_EVENT_OBSERVER_CKPT}" \
  +algorithm.event_sidecar.rgb_student_checkpoint="${ROBOTWIN_EVENT_RGB_STUDENT_CKPT}" \
  +algorithm.event_sidecar.value_lr=1.0e-4 +algorithm.event_sidecar.target_ema_decay=0.995 \
  +algorithm.event_sidecar.proprio_time_delta=5.0 \
  +algorithm.event_branch.num_candidates=4 +algorithm.event_branch.chunk_interval=1 \
  +algorithm.event_branch.horizon=10 +algorithm.event_branch.min_supervision=500 \
  +algorithm.event_branch.influence_lr=1.0e-4 \
  +algorithm.event_credit.granularity=chunk +algorithm.event_credit.max_lambda=0.0 \
  +algorithm.event_credit.min_supervision_for_actor=999999 \
  +algorithm.event_credit.influence_beta=0.02 +algorithm.event_credit.influence_clip=3.0 \
  actor.optim.lr=0.0 actor.optim.value_lr=0.0 actor.model.openpi.noise_method=flow_sde \
  +actor.model.openpi.joint_logprob=false actor.model.openpi.value_after_vlm=false \
  actor.global_batch_size="${ROBOTWIN_GLOBAL_BATCH}" actor.micro_batch_size="${ROBOTWIN_MICRO_BATCH}"
