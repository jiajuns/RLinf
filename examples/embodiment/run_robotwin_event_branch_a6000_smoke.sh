#!/usr/bin/env bash
# One-GPU A6000 preflight for the frozen matched-state branch audit.
#
# This is deliberately a capacity/transport check, not a paper result: it
# uses fewer environments and a smaller batch than the 2/4-GPU protocol.
# It must never be mixed into the formal interaction-budget comparison.
set -euo pipefail

export A6000_ROOT="${A6000_ROOT:-/home/chefmate/Data/pirl_a6000_run}"
export REPO_PATH="${A6000_ROOT}/RLinf-piRL"
export ROBOTWIN_PATH="${A6000_ROOT}/RoboTwin-RLinf_support"
export ROBOTWIN_ASSETS_PATH="${A6000_ROOT}/RoboTwin_v14_dual_gpu_20260831"
export ROBOTWIN_PI05_MODEL="${A6000_ROOT}/RLinf-Pi05-RoboTwin-SFT-adjust_bottle"
export ROBOTWIN_EVENT_OBSERVER_CKPT="${A6000_ROOT}/robotwin_adjust_bottle_event_pretrain_v1_200plus/best.pt"
export ROBOTWIN_EVENT_RGB_STUDENT_CKPT="${A6000_ROOT}/robotwin_adjust_bottle_rgb_student_v1_200plus/best.pt"
# rsync receives this source file at A6000_ROOT/eventvalue_sidecars.pt.
export ROBOTWIN_EVENT_SIDECAR_RESUME="${A6000_ROOT}/eventvalue_sidecars.pt"
export ROBOTWIN_EVENT_DIAGNOSTIC_DIR="${A6000_ROOT}/outputs/robotwin_event_branch_a6000_smoke/raw"
export ROBOTWIN_LOG_PATH="${A6000_ROOT}/outputs/robotwin_event_branch_a6000_smoke"

# All actor/env/rollout components share GPU 0.  Keep this intentionally
# small; increasing it is allowed only after this preflight reports peak VRAM.
export RLINF_COMPONENT_PLACEMENT="${RLINF_COMPONENT_PLACEMENT:-0}"
export RLINF_LOCAL_RAY=1
unset RAY_ADDRESS
# A6000's root filesystem is nearly full; Ray object spilling/logs must stay
# on the 290-GB data disk rather than under /tmp.
export RLINF_RAY_TMPDIR="${RLINF_RAY_TMPDIR:-${A6000_ROOT}/ray_tmp}"
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# RTX 6000 Ada has ample VRAM, but this host has only ~63 GB system RAM.
# A four-environment probe reached Ray's 95% host-memory safety threshold
# during checkpointing.  Two environments leave enough margin for a clean
# one-GPU diagnostic completion; this is never a formal throughput setting.
export ROBOTWIN_TRAIN_ENVS="${ROBOTWIN_TRAIN_ENVS:-2}"
export ROBOTWIN_GLOBAL_BATCH="${ROBOTWIN_GLOBAL_BATCH:-16}"
export ROBOTWIN_MICRO_BATCH="${ROBOTWIN_MICRO_BATCH:-2}"
export ROBOTWIN_BRANCH_COLLECT_EPOCHS="${ROBOTWIN_BRANCH_COLLECT_EPOCHS:-1}"

mkdir -p "${ROBOTWIN_EVENT_DIAGNOSTIC_DIR}" "${ROBOTWIN_LOG_PATH}" "${RLINF_RAY_TMPDIR}"
"${A6000_ROOT}/env_pirl_pi05/bin/python" \
  "${REPO_PATH}/examples/embodiment/train_embodied_agent.py" \
  --config-path config --config-name robotwin_adjust_bottle_ppo_openpi_pi05 \
  runner.max_epochs="${ROBOTWIN_BRANCH_COLLECT_EPOCHS}" \
  runner.logger.log_path="${ROBOTWIN_LOG_PATH}" runner.logger.experiment_name=robotwin_adjust_bottle_branch_a6000_smoke \
  runner.val_check_interval=-1 runner.save_interval="${ROBOTWIN_BRANCH_COLLECT_EPOCHS}" \
  env.train.total_num_envs="${ROBOTWIN_TRAIN_ENVS}" env.train.rollout_epoch=4 env.eval.total_num_envs=1 env.eval.rollout_epoch=1 \
  env.train.assets_path="${ROBOTWIN_ASSETS_PATH}" env.eval.assets_path="${ROBOTWIN_ASSETS_PATH}" \
  env.train.video_cfg.save_video=false env.eval.video_cfg.save_video=false \
  actor.model.model_path="${ROBOTWIN_PI05_MODEL}" rollout.model.model_path="${ROBOTWIN_PI05_MODEL}" \
  +rollout.unnorm_key=adjust_bottle +rollout.collect_transitions=true \
  algorithm.reward_type=chunk_level algorithm.logprob_type=chunk_level \
  algorithm.adv_type=event_smdp_residual algorithm.loss_type=actor_critic \
  +algorithm.event_value_source=learned_sidecar +algorithm.event_boundary_threshold=0.5 \
  +algorithm.event_sidecar.observer_checkpoint="${ROBOTWIN_EVENT_OBSERVER_CKPT}" \
  +algorithm.event_sidecar.rgb_student_checkpoint="${ROBOTWIN_EVENT_RGB_STUDENT_CKPT}" \
  +algorithm.event_sidecar.resume_sidecar_path="${ROBOTWIN_EVENT_SIDECAR_RESUME}" \
  +algorithm.event_sidecar.target_ema_decay=0.995 +algorithm.event_sidecar.proprio_time_delta=50.0 +algorithm.event_sidecar.online_mount_token=2 +algorithm.event_sidecar.bootstrap_on_truncation=false \
  +algorithm.event_sidecar.strict_input_contract=false \
  +algorithm.event_sidecar.value_lr=0.0 \
  +algorithm.event_diagnostics.output_dir="${ROBOTWIN_EVENT_DIAGNOSTIC_DIR}" \
  +algorithm.event_diagnostics.record_only=true \
  +algorithm.event_diagnostics.freeze_sidecars=true \
  +algorithm.event_branch.num_candidates=4 +algorithm.event_branch.chunk_interval=10 \
  +algorithm.event_branch.execution_unit=full_action_chunk +algorithm.event_branch.repeat_primary_candidates=1 +algorithm.event_branch.min_supervision=999999 \
  +algorithm.event_branch.influence_lr=1.0e-4 \
  +algorithm.event_credit.granularity=chunk +algorithm.event_credit.max_lambda=0.0 +algorithm.event_credit.require_ranking_validation=true +algorithm.event_credit.ranking_validation_passed=false \
  +algorithm.event_credit.min_supervision_for_actor=999999 \
  +algorithm.event_credit.influence_beta=0.02 +algorithm.event_credit.influence_clip=3.0 \
  actor.optim.lr=0.0 actor.optim.value_lr=0.0 actor.model.openpi.noise_method=flow_sde \
  +actor.model.openpi.joint_logprob=false actor.model.openpi.value_after_vlm=false \
  actor.global_batch_size="${ROBOTWIN_GLOBAL_BATCH}" actor.micro_batch_size="${ROBOTWIN_MICRO_BATCH}"
