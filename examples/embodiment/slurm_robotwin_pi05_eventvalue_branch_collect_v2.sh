#!/usr/bin/env bash
# Freeze π0.5 SFT and collect matched-state branch labels before Event PPO.
# This job intentionally performs no actor/π0.5 critic learning: it only
# updates V_E, I_xi and its EMA target.  Its branch interactions must be added
# to the final method's total simulator-interaction budget.
#SBATCH --job-name=robotwin_event_branches_v2
#SBATCH --partition=acd_ue
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/robotwin_event_branches_v2_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/robotwin_event_branches_v2_%j.err

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
# Optional prior branch collection.  Resuming is required when collecting the
# calibrated 500+ matched states in multiple four-GPU jobs.
export ROBOTWIN_EVENT_SIDECAR_RESUME="${ROBOTWIN_EVENT_SIDECAR_RESUME:-}"
# If set, persist raw [state,candidate] returns/scores for the influence
# identifiability audit.  This is intentionally opt-in because it is not part
# of ordinary PPO traffic.
export ROBOTWIN_EVENT_DIAGNOSTIC_DIR="${ROBOTWIN_EVENT_DIAGNOSTIC_DIR:-}"
export ROBOTWIN_EVENT_DIAGNOSTIC_RECORD_ONLY="${ROBOTWIN_EVENT_DIAGNOSTIC_RECORD_ONLY:-false}"
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
# Multi-GPU RLinf broadcasts CUDA tensors through IPC.  PyTorch does not
# support that IPC path with expandable segments enabled on the HPC kernel.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:False}"

# Four-GPU equivalent of the upstream eight-GPU recipe: retain the same
# per-GPU rollout and micro-batch load while halving global envs/batch.
export ROBOTWIN_BRANCH_COLLECT_EPOCHS="${ROBOTWIN_BRANCH_COLLECT_EPOCHS:-4}"
export ROBOTWIN_TRAIN_ENVS="${ROBOTWIN_TRAIN_ENVS:-128}"
export ROBOTWIN_GLOBAL_BATCH="${ROBOTWIN_GLOBAL_BATCH:-1024}"
export ROBOTWIN_MICRO_BATCH="${ROBOTWIN_MICRO_BATCH:-32}"
export ROBOTWIN_BRANCH_SAVE_INTERVAL="${ROBOTWIN_BRANCH_SAVE_INTERVAL:-4}"
export ROBOTWIN_EVENT_VALUE_LR="${ROBOTWIN_EVENT_VALUE_LR:-1.0e-4}"
export ROBOTWIN_LOG_PATH="${ROBOTWIN_LOG_PATH:-/data/user/leviccdong/EKSF/outputs/robotwin_event_branches_v2}"
export ROBOTWIN_EXPERIMENT_NAME="${ROBOTWIN_EXPERIMENT_NAME:-robotwin_adjust_bottle_branch_collect_v2}"

cd "${ROBOTWIN_PATH}"
source /share/anaconda3/bin/activate /data/user/leviccdong/EKSF/env_pirl_pi05

sidecar_resume_args=()
if [[ -n "${ROBOTWIN_EVENT_SIDECAR_RESUME}" ]]; then
  sidecar_resume_args=("+algorithm.event_sidecar.resume_sidecar_path=${ROBOTWIN_EVENT_SIDECAR_RESUME}")
fi
diagnostic_args=()
if [[ -n "${ROBOTWIN_EVENT_DIAGNOSTIC_DIR}" ]]; then
  diagnostic_args=(
    "+algorithm.event_diagnostics.output_dir=${ROBOTWIN_EVENT_DIAGNOSTIC_DIR}"
    "+algorithm.event_diagnostics.record_only=${ROBOTWIN_EVENT_DIAGNOSTIC_RECORD_ONLY}"
  )
fi

# This phase is collection/pretraining, not policy evaluation.  Disable eval
# while retaining a final resumable sidecar checkpoint at epoch four.
exec python "${REPO_PATH}/examples/embodiment/train_embodied_agent.py" \
  --config-path config --config-name robotwin_adjust_bottle_ppo_openpi_pi05 \
  runner.max_epochs="${ROBOTWIN_BRANCH_COLLECT_EPOCHS}" \
  runner.logger.log_path="${ROBOTWIN_LOG_PATH}" runner.logger.experiment_name="${ROBOTWIN_EXPERIMENT_NAME}" \
  runner.val_check_interval=-1 runner.save_interval="${ROBOTWIN_BRANCH_SAVE_INTERVAL}" \
  env.train.total_num_envs="${ROBOTWIN_TRAIN_ENVS}" env.train.rollout_epoch=4 env.eval.total_num_envs=128 env.eval.rollout_epoch=1 \
  env.train.assets_path="${ROBOTWIN_ASSETS_PATH}" env.eval.assets_path="${ROBOTWIN_ASSETS_PATH}" \
  env.train.video_cfg.save_video=false env.eval.video_cfg.save_video=false \
  actor.model.model_path="${ROBOTWIN_PI05_MODEL}" rollout.model.model_path="${ROBOTWIN_PI05_MODEL}" \
  +rollout.unnorm_key=adjust_bottle +rollout.collect_transitions=true \
  algorithm.reward_type=chunk_level algorithm.logprob_type=chunk_level \
  algorithm.adv_type=event_smdp_residual algorithm.loss_type=actor_critic \
  +algorithm.event_value_source=learned_sidecar +algorithm.event_boundary_threshold=0.5 \
  +algorithm.event_sidecar.observer_checkpoint="${ROBOTWIN_EVENT_OBSERVER_CKPT}" \
  +algorithm.event_sidecar.rgb_student_checkpoint="${ROBOTWIN_EVENT_RGB_STUDENT_CKPT}" \
  "${sidecar_resume_args[@]}" \
  +algorithm.event_sidecar.target_ema_decay=0.995 \
  +algorithm.event_sidecar.proprio_time_delta=50.0 +algorithm.event_sidecar.online_mount_token=2 +algorithm.event_sidecar.strict_input_contract=true +algorithm.event_sidecar.value_lr="${ROBOTWIN_EVENT_VALUE_LR}" \
  +algorithm.event_branch.num_candidates=4 +algorithm.event_branch.chunk_interval=10 \
  +algorithm.event_branch.execution_unit=full_action_chunk +algorithm.event_branch.min_supervision=500 \
  +algorithm.event_branch.influence_lr=1.0e-4 \
  +algorithm.event_credit.granularity=chunk +algorithm.event_credit.max_lambda=0.0 +algorithm.event_credit.require_ranking_validation=true +algorithm.event_credit.ranking_validation_passed=false \
  +algorithm.event_credit.min_supervision_for_actor=999999 \
  +algorithm.event_credit.influence_beta=0.02 +algorithm.event_credit.influence_clip=3.0 \
  "${diagnostic_args[@]}" \
  actor.optim.lr=0.0 actor.optim.value_lr=0.0 actor.model.openpi.noise_method=flow_sde \
  +actor.model.openpi.joint_logprob=false actor.model.openpi.value_after_vlm=false \
  actor.global_batch_size="${ROBOTWIN_GLOBAL_BATCH}" actor.micro_batch_size="${ROBOTWIN_MICRO_BATCH}"
