# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from pathlib import Path
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.expert import build_expert_model_config
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.event_intervention import EventInfluenceModel, policy_relative_influence
from rlinf.algorithms.event_sidecar_distributed import (
    all_reduce_gradients,
    broadcast_module,
    distributed_ready,
    global_count,
    global_scalar_sum,
)
from rlinf.algorithms.event_value import (
    OnlineEventValueSidecar,
    infer_branch_future_event_values,
    infer_event_sidecar_rollout,
)
from rlinf.config import SupportedModel
from rlinf.data.schema.embodied_types import Trajectory, convert_trajectories_to_batch
from rlinf.data.storage.lerobot import resolve_lerobot_repo_id
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.distributed import (
    all_reduce_dict,
)
from rlinf.utils.metric_utils import (
    CRITIC_EXPLAINED_VARIANCE_KEY,
    append_to_dict,
    compute_critic_explained_variance_from_stats,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
    pop_critic_explained_variance_stats,
)
from rlinf.utils.nested_dict_process import (
    flatten_nested_tensor_time_batch,
    process_nested_dict_for_adv,
    process_nested_dict_for_train,
    put_tensor_device,
    split_dict_to_chunk,
    trim_nested_tensor_time_dim,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    masked_mean,
    reshape_entropy,
)


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self._opd_teacher_model = None
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")
        self._event_sidecar: OnlineEventValueSidecar | None = None
        self._event_value_optimizer: torch.optim.Optimizer | None = None
        self._event_influence_model: EventInfluenceModel | None = None
        self._event_influence_optimizer: torch.optim.Optimizer | None = None
        self._event_branch_supervision_count = 0
        self._event_branch_outcome_count = 0

        self.enable_sft_co_train = cfg.actor.get("enable_sft_co_train", False)
        self.version = 0
        if self.enable_sft_co_train:
            self._build_sft_data_loader()

        # create weight syncer
        weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer")
        self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )
        self.update_epoch = self.cfg.algorithm.get("update_epoch", 1)

        self._sync_weight_comm_options = self.weight_syncer.comm_options

        self._is_weight_sender = self._rank == 0
        self._actor_world_size = self._world_size
        self._rollout_all_ranks = list(
            range(self._component_placement.get_world_size("rollout"))
        )

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        self.setup_model_and_optimizer()

        self._init_online_event_value_sidecar()

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()

    def _init_online_event_value_sidecar(self) -> None:
        """Load the frozen online-capable Observer and its trainable ``V_E``.

        The sidecar is opt-in.  Ordinary πRL / GAE configurations therefore
        instantiate exactly the upstream actor and retain a clean baseline.
        Its checkpoints are produced from the task-matched SAM-teacher cache;
        no oracle labels or Zarr lookup are used during PPO rollout.
        """
        source = self.cfg.algorithm.get("event_value_source", None)
        if source != "learned_sidecar":
            return
        if self.cfg.algorithm.adv_type not in (
            "event_smdp_temporal",
            "event_smdp_interventional",
            "event_smdp_residual",
        ):
            raise ValueError("algorithm.event_value_source=learned_sidecar needs an Event-SMDP advantage")
        sidecar_cfg = self.cfg.algorithm.get("event_sidecar", None)
        if sidecar_cfg is None:
            raise ValueError("learned_sidecar requires algorithm.event_sidecar checkpoint paths")
        observer_checkpoint = sidecar_cfg.get("observer_checkpoint", None)
        rgb_student_checkpoint = sidecar_cfg.get("rgb_student_checkpoint", None)
        if not observer_checkpoint or not rgb_student_checkpoint:
            raise ValueError("event_sidecar needs observer_checkpoint and rgb_student_checkpoint")
        # RLinf stores the local device as an integer rank in this worker,
        # whereas ``torch.load(map_location=...)`` requires a device-like
        # value.  Keep the conversion local to the auxiliary Event sidecar so
        # the upstream actor device convention remains untouched.
        sidecar_device = (
            torch.device(f"cuda:{self.device}")
            if isinstance(self.device, int)
            else torch.device(self.device)
        )
        # One observer output represents one *complete* π0.5 action chunk.
        # This contract is checkpointed by both offline trainers and checked
        # at load time; it prevents a raw-control-step proprio derivative
        # from being silently used for a chunk-boundary online sequence.
        proprio_time_delta = float(
            sidecar_cfg.get("proprio_time_delta", self.cfg.actor.model.num_action_chunks)
        )
        online_mount_token = int(sidecar_cfg.get("online_mount_token", 2))
        self._event_sidecar = OnlineEventValueSidecar.from_checkpoints(
            observer_checkpoint,
            rgb_student_checkpoint,
            device=sidecar_device,
            proprio_time_delta=proprio_time_delta,
            online_mount_token=online_mount_token,
            require_input_contract=bool(sidecar_cfg.get("strict_input_contract", True)),
        )
        target_ema = float(sidecar_cfg.get("target_ema_decay", 0.995))
        if not 0.0 <= target_ema < 1.0:
            raise ValueError("algorithm.event_sidecar.target_ema_decay must be in [0, 1)")
        self._event_sidecar.target_ema_decay = target_ema
        self._event_value_optimizer = torch.optim.AdamW(
            self._event_sidecar.event_value.parameters(),
            lr=float(sidecar_cfg.get("value_lr", 1.0e-4)),
            weight_decay=float(sidecar_cfg.get("weight_decay", 0.0)),
        )
        branch_cfg = self.cfg.algorithm.get("event_branch", {})
        if int(branch_cfg.get("num_candidates", 0)):
            representation_dim = self._event_sidecar.event_value.value[1].in_features
            credit_cfg = self.cfg.algorithm.get("event_credit", {})
            granularity = str(credit_cfg.get("granularity", "action"))
            if granularity not in ("action", "chunk"):
                raise ValueError("algorithm.event_credit.granularity must be 'action' or 'chunk'")
            influence_action_dim = self.cfg.actor.model.action_dim
            if granularity == "chunk":
                # A same-state branch labels the sampled Flow action *chunk*.
                # Feeding a single token would make its target non-identifiable.
                influence_action_dim *= self.cfg.actor.model.num_action_chunks
            self._event_influence_model = EventInfluenceModel(
                representation_dim,
                influence_action_dim,
                hidden_dim=int(branch_cfg.get("influence_hidden_dim", 256)),
            ).to(sidecar_device)
            self._event_influence_optimizer = torch.optim.AdamW(
                self._event_influence_model.parameters(),
                lr=float(branch_cfg.get("influence_lr", 1.0e-4)),
                weight_decay=float(branch_cfg.get("influence_weight_decay", 0.0)),
            )
        resume_path = sidecar_cfg.get("resume_sidecar_path", None)
        if resume_path:
            payload = torch.load(Path(resume_path), map_location=sidecar_device, weights_only=False)
            if payload.get("format") not in {"eventvalue_rl_sidecars_v1", "eventvalue_rl_sidecars_v2"}:
                raise ValueError("event sidecar resume checkpoint has an unsupported format")
            if payload.get("time_unit", "full_action_chunk") != "full_action_chunk":
                raise ValueError("cannot resume a sidecar trained on a non-chunk time unit")
            if payload.get("proprio_time_delta") is not None and float(payload["proprio_time_delta"]) != float(
                self._event_sidecar.proprio_time_delta
            ):
                raise ValueError("resume sidecar proprio_time_delta differs from current online contract")
            if payload.get("online_mount_token") is not None and int(payload["online_mount_token"]) != int(
                self._event_sidecar.online_mount_token
            ):
                raise ValueError("resume sidecar mount token differs from current online contract")
            self._event_sidecar.event_value.load_state_dict(payload["event_value"])
            self._event_sidecar.target_event_value.load_state_dict(
                payload.get("target_event_value", payload["event_value"])
            )
            self._event_branch_supervision_count = int(payload.get("branch_supervision_count", 0))
            self._event_branch_outcome_count = int(payload.get("branch_outcome_count", 0))
            if self._event_influence_model is not None and payload.get("influence_model") is not None:
                self._event_influence_model.load_state_dict(payload["influence_model"])
            if self._event_influence_optimizer is not None and payload.get("influence_optimizer") is not None:
                self._event_influence_optimizer.load_state_dict(payload["influence_optimizer"])
            if payload.get("event_value_optimizer") is not None:
                self._event_value_optimizer.load_state_dict(payload["event_value_optimizer"])
                self._move_optimizer_state_to_device(self._event_value_optimizer, sidecar_device)
            if self._event_influence_optimizer is not None:
                self._move_optimizer_state_to_device(self._event_influence_optimizer, sidecar_device)
        # All FSDP ranks must start the sidecars identically.  In particular
        # the Influence MLP is freshly randomized on every rank otherwise.
        self._synchronize_event_sidecars_from_rank0()

    @staticmethod
    def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        """Move deserialised Adam moments back to the sidecar device."""
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    def _event_sidecar_distributed(self) -> bool:
        return distributed_ready(self._world_size)

    def _synchronize_event_sidecars_from_rank0(self) -> None:
        """Broadcast auxiliary modules after init/resume, never actor weights."""
        if not self._event_sidecar_distributed() or self._event_sidecar is None:
            return
        modules: list[nn.Module] = [self._event_sidecar.event_value, self._event_sidecar.target_event_value]
        if self._event_influence_model is not None:
            modules.append(self._event_influence_model)
        for module in modules:
            broadcast_module(module, world_size=self._world_size, src=0)

    def _all_reduce_auxiliary_gradients(self, module: nn.Module, *, normalizer: int | None = None) -> None:
        """Synchronize sidecar gradients with an explicit global normalizer.

        For masked auxiliary losses every rank backpropagates its *local sum*;
        ranks without examples backpropagate a graph-connected zero.  Dividing
        the summed gradient by global valid-example count therefore matches a
        true global masked mean and keeps all Adam steps identical.
        """
        all_reduce_gradients(module, world_size=self._world_size, normalizer=normalizer)

    def _global_auxiliary_count(self, local_count: int, device: torch.device) -> int:
        return global_count(local_count, world_size=self._world_size, device=device)

    def _global_auxiliary_scalar(self, local_value: torch.Tensor) -> torch.Tensor:
        """Return a detached summed scalar on every rank for metrics only."""
        return global_scalar_sum(local_value, world_size=self._world_size)

    def _event_action_tensor(
        self, *, chunks: int, batch: int, action_chunk: int, device: torch.device
    ) -> torch.Tensor:
        """Return actions at the same granularity as branch interventions."""
        actions = self.rollout_batch.get("actions")
        if actions is None:
            raise RuntimeError("Event Influence Model needs recorded executed actions")
        action_dim = self.cfg.actor.model.action_dim
        if actions.shape[:2] != (chunks, batch):
            raise RuntimeError("recorded actions are not aligned to the event rollout")
        if actions.shape[-1] != action_chunk * action_dim:
            raise RuntimeError(
                "Event Influence Model expects flattened executed actions with "
                f"width {action_chunk * action_dim}, got {actions.shape[-1]}"
            )
        action_chunks = actions.to(device).reshape(chunks, batch, action_chunk, action_dim)
        granularity = str(self.cfg.algorithm.get("event_credit", {}).get("granularity", "action"))
        if granularity == "chunk":
            return action_chunks.permute(1, 0, 2, 3).reshape(batch, chunks, action_chunk * action_dim)
        if granularity != "action":
            raise ValueError("event credit granularity must be 'action' or 'chunk'")
        return action_chunks.permute(1, 0, 2, 3).reshape(batch, chunks * action_chunk, action_dim)

    def _policy_relative_chunk_influence(
        self,
        representations: torch.Tensor,
        actions: torch.Tensor,
        *,
        chunks: int,
        batch: int,
        action_chunk: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return ``f(z,a)-E_{a'~pi}[f(z,a')]`` on the chunk clock.

        The sidecar is trained with a state-centred objective.  Its raw score
        is therefore not identifiable across two different states: an
        arbitrary state-only offset leaves the training objective unchanged.
        The rollout worker transports non-executed Flow-SDE samples from the
        *same observation* so PPO can use the only identified quantity.
        They are model-inference samples, not simulator branches, and are
        deliberately excluded from interaction accounting.
        """
        references, _ = self._aligned_policy_reference_actions(
            self.rollout_batch.get("influence_reference_actions"), chunks=chunks, batch=batch
        )
        if references is None:
            return torch.zeros((chunks, batch), device=device, dtype=actions.dtype)
        action_dim = self.cfg.actor.model.action_dim
        if references.shape[2] < 2 or references.shape[3:] != (action_chunk, action_dim):
            raise RuntimeError("policy-relative Influence reference chunks have an invalid shape")
        reference_actions = references.to(device).permute(1, 0, 2, 3, 4).reshape(
            batch, chunks, references.shape[2], action_chunk * action_dim
        )
        relative = policy_relative_influence(
            self._event_influence_model,
            representations,
            actions,
            reference_actions,
        )
        return relative.transpose(0, 1)

    def _aligned_policy_reference_actions(
        self, references: torch.Tensor | None, *, chunks: int, batch: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Align delayed rollout reference chunks to the actor's chunk clock.

        The rollout pipeline can omit the final bootstrap-only observation:
        that observation has no policy action/reference but the branch sidecar
        owns a terminal padding chunk.  Pad only this documented one-chunk
        tail and propagate an explicit availability mask; never fabricate a
        reference score for that tail.
        """
        if references is None:
            return None, None
        if references.ndim != 5 or references.shape[1] != batch:
            raise RuntimeError(
                "policy-relative Influence references must be "
                "[chunks,batch,candidates,action_chunk,action_dim]; "
                f"got {tuple(references.shape)} for chunks={chunks}, batch={batch}"
            )
        available = torch.ones(references.shape[:2], dtype=torch.bool, device=references.device)
        if references.shape[0] == chunks:
            return references, available
        if references.shape[0] == chunks - 1:
            tail = torch.zeros_like(references[:1])
            return torch.cat((references, tail), dim=0), torch.cat(
                (available, torch.zeros_like(available[:1])), dim=0
            )
        raise RuntimeError(
            "policy-relative Influence reference clock is not compatible with branch clock: "
            f"got {tuple(references.shape)}, expected {chunks} or {chunks - 1} chunks"
        )

    def _event_credit_mix_lambda(self) -> float:
        """Return a conservative, branch-gated Event residual coefficient.

        This makes ``lambda=0`` a literal πRL/GAE control until the configured
        amount of matched-state evidence exists.  The schedule is driven by
        optimizer updates rather than wall clock time so resumed Slurm jobs do
        not accidentally skip the safety warm-up.
        """
        cfg = self.cfg.algorithm.get("event_credit", {})
        # Matched-state labels alone are not enough: a small MSE can be the
        # zero predictor when branch returns are tied.  A held-out diagnostic
        # must explicitly certify ranking before Event credit can affect PPO.
        if bool(cfg.get("require_ranking_validation", True)) and not bool(
            cfg.get("ranking_validation_passed", False)
        ):
            return 0.0
        required_labels = int(cfg.get("min_supervision_for_actor", 0))
        if self._event_branch_supervision_count < required_labels:
            return 0.0
        maximum = float(cfg.get("max_lambda", 0.0))
        if not 0.0 <= maximum <= 1.0:
            raise ValueError("algorithm.event_credit.max_lambda must be in [0, 1]")
        if maximum > 0.0:
            if str(cfg.get("granularity", "action")) != "chunk":
                raise ValueError("learned policy-relative Influence currently requires chunk granularity")
            if int(cfg.get("reference_candidates", 0)) < 2:
                raise ValueError(
                    "Event credit with a state-centered Influence score requires "
                    "event_credit.reference_candidates >= 2"
                )
        warmup = int(cfg.get("warmup_optimizer_steps", 0))
        ramp = int(cfg.get("ramp_optimizer_steps", 1))
        if self.optimizer_steps < warmup:
            return 0.0
        return maximum * min(1.0, (self.optimizer_steps - warmup) / max(ramp, 1))

    def _write_event_branch_diagnostics(
        self,
        *,
        sidecar_rollout,
        branch_returns: torch.Tensor,
        predicted_scores: torch.Tensor,
        branch_mask: torch.Tensor,
        branch_valid: torch.Tensor,
        branch_rewards: torch.Tensor,
        branch_bootstrap_values: torch.Tensor,
        branch_discounted_bootstrap: torch.Tensor,
        branch_discount_units: torch.Tensor,
        branch_bootstrap_allowed: torch.Tensor,
        branch_horizons: torch.Tensor,
        branch_requested_steps: torch.Tensor,
        branch_terminations: torch.Tensor,
        branch_truncations: torch.Tensor,
        branch_state_ids: torch.Tensor,
        branch_actions: torch.Tensor,
        state_representations: torch.Tensor,
        policy_reference_actions: torch.Tensor | None,
        policy_reference_mask: torch.Tensor | None,
    ) -> None:
        """Persist raw matched-state branch diagnostics for offline auditing.

        The old sidecar checkpoints intentionally contain only model state and
        aggregate counts.  That is insufficient for testing whether a small
        MSE is meaningful: we need the four true returns and four predicted
        scores *per state*.  One compressed file per FSDP rank/update avoids
        cross-rank writes while keeping the artifact compact (no RGB tensors).
        """
        cfg = self.cfg.algorithm.get("event_diagnostics", {})
        output_dir = cfg.get("output_dir", None)
        if not output_dir:
            return
        if branch_returns.shape != predicted_scores.shape:
            raise RuntimeError("diagnostic prediction and branch-return shapes must match")
        if branch_mask.shape != branch_returns.shape[:2]:
            raise RuntimeError("diagnostic branch mask is not aligned to branch returns")
        if branch_valid.shape != branch_returns.shape:
            raise RuntimeError("diagnostic valid-sample mask is not aligned to branch returns")
        if branch_state_ids.shape[:2] != branch_returns.shape[:2] or branch_state_ids.shape[-1] < 2:
            raise RuntimeError("diagnostic cloned-state identity must include seed and elapsed-step columns")
        if state_representations.shape[:2] != branch_returns.shape[:2] or state_representations.ndim != 3:
            raise RuntimeError("diagnostic Event representations are not aligned to branch returns")
        if policy_reference_actions is not None:
            if policy_reference_actions.ndim != 5 or policy_reference_actions.shape[:2] != branch_returns.shape[:2]:
                raise RuntimeError(
                    "policy-reference actions must be [chunks,batch,candidates,action_chunk,action_dim]; "
                    f"got {tuple(policy_reference_actions.shape)}, expected prefix "
                    f"{tuple(branch_returns.shape[:2])}"
                )
            if policy_reference_mask is None or policy_reference_mask.shape != branch_returns.shape[:2]:
                raise RuntimeError("policy-reference availability mask is not aligned to branch returns")
        selected = branch_mask.bool()
        if not bool(selected.any()):
            return

        event_ids = sidecar_rollout.event_ids
        if event_ids.ndim != 3:
            raise RuntimeError("Event diagnostics expect event IDs shaped [chunks,batch,action_chunk]")
        # ``infer_event_sidecar_rollout`` already returns [C,B,K], matching
        # the branch-return layout.  Token zero identifies the event for a
        # chunk-level controlled intervention.  Do not transpose here: an
        # earlier diagnostic-only implementation incorrectly assumed a
        # batch-major layout, which only surfaced when C != B.
        event_ids = event_ids[..., 0]
        if event_ids.shape != branch_returns.shape[:2]:
            raise RuntimeError("event IDs are not aligned to diagnostic branches")

        branch_success = self.rollout_batch.get("branch_success")
        success_available = branch_success is not None
        if branch_success is None:
            branch_success = torch.zeros_like(branch_returns, dtype=torch.bool)
        else:
            branch_success = branch_success.to(branch_returns.device).bool()
            if branch_success.shape != branch_returns.shape:
                raise RuntimeError("branch success flags are not aligned to branch returns")
        policy_versions = self.rollout_batch.get("versions")
        if policy_versions is None:
            policy_versions = torch.full_like(event_ids, -1, dtype=torch.long)
        else:
            if policy_versions.shape[:2] != event_ids.shape:
                raise RuntimeError("policy versions are not aligned to diagnostic branches")
            policy_versions = (
                policy_versions[..., 0] if policy_versions.ndim == 3 else policy_versions
            ).to(event_ids.device)

        output = Path(str(output_dir))
        output.mkdir(parents=True, exist_ok=True)
        # Include the cumulative real-label count and FSDP rank to make files
        # unique even after Slurm resume/requeue.
        path = output / (
            f"branch_diag_rank{self._rank}_opt{self.optimizer_steps}_"
            f"seen{self._event_branch_supervision_count}.npz"
        )
        # ``record_only`` collection is commonly restarted from the same
        # frozen sidecar, so rank/optimizer/count alone are not unique across
        # independent invocations.  Preserve every collection rather than
        # silently overwriting an earlier held-out diagnostic batch.
        if path.exists():
            stem, suffix = path.stem, path.suffix
            retry = 1
            while path.exists():
                path = output / f"{stem}_retry{retry}{suffix}"
                retry += 1
        np.savez_compressed(
            path,
            branch_returns=branch_returns[selected].detach().float().cpu().numpy(),
            predicted_scores=predicted_scores[selected].detach().float().cpu().numpy(),
            candidate_actions=branch_actions[selected].detach().float().cpu().numpy(),
            # These non-executed Flow-SDE samples are retained solely for
            # reference-mean stability audits (2/4/8 samples).  They are not
            # simulator data and never become policy/Observer inputs.
            policy_reference_actions=(
                policy_reference_actions[selected].detach().float().cpu().numpy()
                if policy_reference_actions is not None
                else np.empty((int(selected.sum().item()), 0), dtype=np.float32)
            ),
            policy_reference_available=(
                policy_reference_mask[selected].detach().cpu().numpy()
                if policy_reference_mask is not None
                else np.zeros((int(selected.sum().item()),), dtype=np.bool_)
            ),
            candidate_valid_mask=branch_valid[selected].detach().cpu().numpy(),
            branch_rewards=branch_rewards[selected].detach().float().cpu().numpy(),
            branch_bootstrap_values=branch_bootstrap_values[selected].detach().float().cpu().numpy(),
            branch_discounted_bootstrap=branch_discounted_bootstrap[selected].detach().float().cpu().numpy(),
            bootstrap_discount_units=branch_discount_units[selected].detach().cpu().numpy(),
            bootstrap_allowed_mask=branch_bootstrap_allowed[selected].detach().cpu().numpy(),
            actual_duration_steps=branch_horizons[selected].detach().cpu().numpy(),
            requested_duration_steps=branch_requested_steps[selected].detach().cpu().numpy(),
            branch_terminations=branch_terminations[selected].detach().cpu().numpy(),
            branch_truncations=branch_truncations[selected].detach().cpu().numpy(),
            state_ids=branch_state_ids[selected].detach().cpu().numpy(),
            # A frozen online-capable Event representation is sufficient to
            # train/validate I_xi offline; raw RGB or simulator oracle state
            # is deliberately not written into branch diagnostics.
            state_representations=state_representations[selected].detach().float().cpu().numpy(),
            # Files contain only selected valid branch states; retain an
            # explicit row mask so downstream readers need not infer that
            # contract from filename conventions.
            supervision_mask=np.ones(int(selected.sum().item()), dtype=np.bool_),
            event_ids=event_ids[selected].detach().cpu().numpy(),
            branch_success=branch_success[selected].detach().cpu().numpy(),
            success_available=np.asarray(success_available, dtype=np.bool_),
            gamma=np.asarray(float(self.cfg.algorithm.get("gamma", 1.0)), dtype=np.float32),
            actor_rank=np.asarray(self._rank, dtype=np.int64),
            optimizer_step=np.asarray(self.optimizer_steps, dtype=np.int64),
            policy_versions=policy_versions[selected].detach().cpu().numpy(),
            diagnostic_run_id=np.asarray(str(cfg.get("run_id", "unspecified"))),
        )

    def _update_influence_from_branches(
        self,
        sidecar_rollout,
        sidecar_curr_obs: dict[str, torch.Tensor],
        *,
        chunks: int,
        batch: int,
        action_chunk: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Train ``I_ξ`` on real branch labels and return PPO influences.

        Branch outcomes supervise a state-centered candidate score.  PPO never
        consumes that raw score: it consumes the score relative to same-state,
        non-executed policy samples, which removes the unidentifiable
        state-common offset.  Before enough real labels exist, exact zeros
        select uniform Event-SMDP credit rather than random causal scores.
        """
        if self._event_influence_model is None or self._event_influence_optimizer is None:
            return torch.zeros(
                (chunks, batch, action_chunk), device=self.rollout_batch["rewards"].device
            ), None
        actions = self._event_action_tensor(
            chunks=chunks, batch=batch, action_chunk=action_chunk, device=device
        )
        granularity = str(self.cfg.algorithm.get("event_credit", {}).get("granularity", "action"))
        if granularity == "chunk":
            representations = sidecar_rollout.chunk_output.representation[:, :-1]
            chunk_prediction = self._event_influence_model(representations, actions).transpose(0, 1)
            # Keep the native [C,B,K] sidecar interface.  The chunk-level
            # advantage preprocessor selects token zero, so the repeated
            # entries are never interpreted as independently supervised.
            prediction = chunk_prediction.unsqueeze(-1).expand(-1, -1, action_chunk)
        else:
            representations = sidecar_rollout.action_representation[:, :-1]
            prediction = self._event_influence_model(representations, actions).reshape(
                batch, chunks, action_chunk
            ).permute(1, 0, 2)
        # Every rank starts with a graph-connected zero.  This is essential:
        # a rank without selected branch states must still participate in the
        # same gradient all-reduce as ranks that have supervision.
        local_loss_sum = chunk_prediction.sum() * 0.0 if granularity == "chunk" else prediction.sum() * 0.0
        local_supervision_count = 0
        local_outcome_count = 0
        influence_loss = None
        record_only = bool(self.cfg.algorithm.get("event_diagnostics", {}).get("record_only", False))
        branch_mask = self.rollout_batch.get("branch_mask")
        required_branch_fields = (
            "branch_rewards",
            "branch_horizons",
            "branch_requested_steps",
            "branch_valid",
            "branch_terminations",
            "branch_truncations",
            "branch_state_ids",
            "branch_main_images",
            "branch_wrist_images",
        )
        if branch_mask is not None and all(
            self.rollout_batch.get(field) is not None for field in required_branch_fields
        ):
            mask = branch_mask.to(device).bool()
            if mask.shape != (chunks, batch):
                raise RuntimeError("branch supervision mask is not aligned to rollout chunks")
            if bool(mask.any()):
                future_values = infer_branch_future_event_values(
                    self._event_sidecar,
                    sidecar_curr_obs,
                    self.rollout_batch["branch_main_images"].to(device),
                    self.rollout_batch["branch_wrist_images"].to(device),
                    self.rollout_batch.get("branch_measured_state16", None).to(device)
                    if self.rollout_batch.get("branch_measured_state16", None) is not None
                    else None,
                )
                branch_rewards = self.rollout_batch["branch_rewards"].to(device)
                branch_horizons = self.rollout_batch["branch_horizons"].to(device)
                branch_requested_steps = self.rollout_batch["branch_requested_steps"].to(device)
                branch_valid = self.rollout_batch["branch_valid"].to(device).bool()
                branch_terminations = self.rollout_batch["branch_terminations"].to(device).bool()
                branch_truncations = self.rollout_batch["branch_truncations"].to(device).bool()
                branch_state_ids = self.rollout_batch["branch_state_ids"].to(device)
                if branch_rewards.shape != future_values.shape or branch_horizons.shape != future_values.shape:
                    raise RuntimeError("branch return tensors are inconsistent")
                if branch_requested_steps.shape != branch_rewards.shape or branch_valid.shape != branch_rewards.shape:
                    raise RuntimeError("branch duration/valid masks are inconsistent")
                bootstrap_on_truncation = bool(
                    self.cfg.algorithm.get("event_sidecar", {}).get("bootstrap_on_truncation", False)
                )
                branch_bootstrap_allowed = ~branch_terminations & (
                    torch.ones_like(branch_truncations, dtype=torch.bool)
                    if bootstrap_on_truncation
                    else ~branch_truncations
                )
                # The SMDP/PPO clock is chunks, not raw controller ticks:
                # one candidate executes one complete action chunk and hence
                # advances the bootstrap by exactly one RL time unit.  Raw
                # action-step duration remains recorded for simulator audit,
                # but must not turn a chunk-level gamma into gamma**50.
                branch_discount_units = torch.ones_like(branch_horizons)
                discount = torch.pow(
                    torch.as_tensor(float(self.cfg.algorithm.get("gamma", 1.0)), device=device),
                    branch_discount_units,
                )
                # Terminations never bootstrap.  Truncation follows the same
                # explicit rule as the Event Value/PPO configuration, and the
                # resulting mask is persisted for audit.
                branch_bootstrap = torch.where(
                    branch_bootstrap_allowed, future_values, torch.zeros_like(future_values)
                )
                discounted_bootstrap = discount * branch_bootstrap
                branch_returns = branch_rewards + discounted_bootstrap
                # Candidate 0 is the executed Flow-SDE trajectory.  Centering
                # across matched candidates preserves beneficial and harmful
                # signs in the signed Event-SMDP allocator.
                valid_count = branch_valid.sum(dim=-1)
                matched_mean = (branch_returns * branch_valid).sum(dim=-1) / valid_count.clamp_min(1)
                target = branch_returns[..., 0] - matched_mean
                supervision_mask = mask & branch_valid[..., 0] & (valid_count >= 2)
                direct_prediction = chunk_prediction if granularity == "chunk" else prediction[..., 0]
                if bool(supervision_mask.any()):
                    local_loss_sum = (direct_prediction - target).square().masked_select(supervision_mask).sum()
                    local_supervision_count = int(supervision_mask.sum().item())
                    local_outcome_count = int(branch_valid[mask].sum().item())
                branch_actions = self.rollout_batch.get("branch_actions")
                if branch_actions is not None and granularity == "chunk":
                    # Every candidate is scored with the same state/event
                    # representation and its own sampled Flow-SDE chunk.  The
                    # score matrix trains the same state-centred objective as
                    # the offline ranking diagnostics.
                    candidate_actions = branch_actions.to(device)
                    expected_shape = (chunks, batch, branch_returns.shape[-1], action_chunk)
                    if candidate_actions.shape[:4] != expected_shape:
                        raise RuntimeError(
                            "branch action candidates are not aligned to Event diagnostics: "
                            f"got {candidate_actions.shape}, expected prefix {expected_shape}"
                        )
                    candidate_actions = candidate_actions.permute(1, 0, 2, 3, 4).flatten(start_dim=-2)
                    candidate_representations = representations.unsqueeze(2).expand(
                        -1, -1, candidate_actions.shape[2], -1
                    )
                    candidate_scores = self._event_influence_model(
                        candidate_representations, candidate_actions
                    ).permute(1, 0, 2)
                    # Each Flow-SDE candidate is a real intervention outcome
                    # from the *same* restored state.  Training only on
                    # candidate zero leaves three quarters of that costly
                    # supervision unused and cannot validate a candidate
                    # ranking.  Regress every valid candidate to its centered
                    # matched-state return; candidate zero remains the PPO
                    # action queried by ``prediction`` below.
                    candidate_target = branch_returns - matched_mean.unsqueeze(-1)
                    candidate_supervision_mask = (
                        mask.unsqueeze(-1)
                        & branch_valid
                        & (valid_count.unsqueeze(-1) >= 2)
                    )
                    if bool(candidate_supervision_mask.any()):
                        # Do not regress raw scores to centred labels.  The
                        # objective identifies only candidate differences; a
                        # per-state score offset is intentionally free.
                        score_count = candidate_supervision_mask.sum(dim=-1, keepdim=True)
                        score_mean = (
                            (candidate_scores * candidate_supervision_mask).sum(dim=-1, keepdim=True)
                            / score_count.clamp_min(1)
                        )
                        candidate_scores_centered = candidate_scores - score_mean
                        local_loss_sum = (candidate_scores_centered - candidate_target).square().masked_select(
                            candidate_supervision_mask
                        ).sum()
                        local_supervision_count = int(candidate_supervision_mask.sum().item())
                        local_outcome_count = local_supervision_count
                    policy_reference_actions, policy_reference_mask = self._aligned_policy_reference_actions(
                        self.rollout_batch.get("influence_reference_actions"), chunks=chunks, batch=batch
                    )
                    self._write_event_branch_diagnostics(
                        sidecar_rollout=sidecar_rollout,
                        branch_returns=branch_returns,
                        predicted_scores=candidate_scores,
                        branch_mask=mask,
                        branch_valid=branch_valid,
                        branch_rewards=branch_rewards,
                        branch_bootstrap_values=branch_bootstrap,
                        branch_discounted_bootstrap=discounted_bootstrap,
                        branch_discount_units=branch_discount_units,
                        branch_bootstrap_allowed=branch_bootstrap_allowed,
                        branch_horizons=branch_horizons,
                        branch_requested_steps=branch_requested_steps,
                        branch_terminations=branch_terminations,
                        branch_truncations=branch_truncations,
                        branch_state_ids=branch_state_ids,
                        branch_actions=self.rollout_batch["branch_actions"].to(device),
                        state_representations=representations.permute(1, 0, 2),
                        policy_reference_actions=(
                            policy_reference_actions.to(device)
                            if policy_reference_actions is not None
                            else None
                        ),
                        policy_reference_mask=(
                            policy_reference_mask.to(device)
                            if policy_reference_mask is not None
                            else None
                        ),
                    )
                elif self.cfg.algorithm.get("event_diagnostics", {}).get("output_dir", None):
                    raise RuntimeError(
                        "Event diagnostics require chunk granularity and transported branch actions"
                    )

        global_supervision_count = self._global_auxiliary_count(local_supervision_count, device)
        global_outcome_count = self._global_auxiliary_count(local_outcome_count, device)
        global_loss_sum = self._global_auxiliary_scalar(local_loss_sum)
        if global_supervision_count:
            influence_loss = global_loss_sum / global_supervision_count
            if not record_only:
                self._event_influence_optimizer.zero_grad(set_to_none=True)
                local_loss_sum.backward()
                self._all_reduce_auxiliary_gradients(
                    self._event_influence_model, normalizer=global_supervision_count
                )
                torch.nn.utils.clip_grad_norm_(self._event_influence_model.parameters(), 1.0)
                self._event_influence_optimizer.step()
                # All ranks receive the same global increments and thus retain
                # an identical branch-gating schedule after resume.
                self._event_branch_supervision_count += global_supervision_count
                self._event_branch_outcome_count += global_outcome_count

        minimum = int(self.cfg.algorithm.get("event_branch", {}).get("min_supervision", 1))
        if self._event_branch_supervision_count < minimum:
            allocated = torch.zeros_like(prediction)
        elif granularity == "chunk":
            # ``prediction`` carries an arbitrary state-only intercept under
            # state-centred training and must never be used directly as credit.
            relative_chunk = self._policy_relative_chunk_influence(
                representations,
                actions,
                chunks=chunks,
                batch=batch,
                action_chunk=action_chunk,
                device=device,
            )
            allocated = relative_chunk.unsqueeze(-1).expand(-1, -1, action_chunk).detach()
        else:
            # Action-level branch labels are not identifiable from whole
            # action chunks.  Keep this legacy path inert until a matching
            # action-level intervention/reference protocol exists.
            allocated = torch.zeros_like(prediction)
        return allocated.to(self.rollout_batch["rewards"].device), influence_loss.detach() if influence_loss is not None else None

    def _populate_learned_event_sidecar(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Attach learned online event values/IDs and update ``V_E`` once.

        The event representation is evaluated from RGB stored in the current
        rollout, frozen before PPO.  We then make one detached TD/SMDP update
        of the separate critic from those current-policy returns.  Updating it
        here, rather than in the FSDP actor optimizer, makes the policy/value
        parameter boundary explicit and prevents gradients entering π0.5.
        """
        if self._event_sidecar is None or self._event_value_optimizer is None:
            raise RuntimeError("learned Event Value sidecar was not initialized")
        curr_obs = self.rollout_batch.get("curr_obs")
        next_obs = self.rollout_batch.get("next_obs")
        if not curr_obs or not next_obs:
            raise RuntimeError(
                "learned Event Value needs rollout RGB. Set rollout.collect_transitions=true "
                "for the Event-SMDP configuration."
            )
        sidecar_device = next(self._event_sidecar.event_value.parameters()).device
        sidecar_curr_obs = {
            key: value.to(sidecar_device, non_blocking=True)
            for key, value in curr_obs.items()
            if isinstance(value, torch.Tensor)
        }
        sidecar_next_obs = {
            key: value.to(sidecar_device, non_blocking=True)
            for key, value in next_obs.items()
            if isinstance(value, torch.Tensor)
        }
        sidecar_rollout = infer_event_sidecar_rollout(
            self._event_sidecar,
            sidecar_curr_obs,
            sidecar_next_obs,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            boundary_threshold=float(
                self.cfg.algorithm.get("event_boundary_threshold", 0.5)
            ),
        )
        rollout_device = self.rollout_batch["rewards"].device
        self.rollout_batch["event_ids"] = sidecar_rollout.event_ids.to(rollout_device)
        self.rollout_batch["event_values"] = sidecar_rollout.event_values.to(rollout_device)

        rewards = self.rollout_batch["rewards"]
        terminations = self.rollout_batch.get("terminations")
        truncations = self.rollout_batch.get("truncations")
        if terminations is None or truncations is None:
            raise RuntimeError("Event Value needs separate termination and truncation tensors")
        chunks, batch, action_chunk = rewards.shape
        influence, influence_loss = self._update_influence_from_branches(
            sidecar_rollout,
            sidecar_curr_obs,
            chunks=chunks,
            batch=batch,
            action_chunk=action_chunk,
            device=sidecar_device,
        )
        self.rollout_batch["intervention_influence"] = influence.to(
            dtype=self.rollout_batch["event_values"].dtype
        )
        # PPO has already declared a complete sampled π0.5 action chunk to be
        # one reward/terminal/GAE transition (reward_type=chunk_level).  V_E
        # must use that identical clock, not expand it back into 50 pseudo
        # transitions.  This makes the SMDP duration and branch gamma exponent
        # directly comparable.
        chunk_rewards = rewards.to(sidecar_device).sum(dim=-1)
        chunk_terminations = terminations.to(sidecar_device).max(dim=-1).values
        chunk_truncations = truncations.to(sidecar_device).max(dim=-1).values
        chunk_event_ids = sidecar_rollout.event_ids[..., 0]
        raw_loss_mask = self.rollout_batch.get("loss_mask")
        if raw_loss_mask is None:
            chunk_valid_mask = torch.ones_like(chunk_rewards, dtype=torch.bool)
        else:
            raw_loss_mask = raw_loss_mask.to(sidecar_device)
            if raw_loss_mask.shape[:2] != (chunks, batch):
                raise RuntimeError("loss_mask is not aligned to chunk Event Value rollout")
            chunk_valid_mask = raw_loss_mask.bool().any(dim=-1) if raw_loss_mask.ndim == 3 else raw_loss_mask.bool()
        freeze_sidecars = bool(self.cfg.algorithm.get("event_diagnostics", {}).get("freeze_sidecars", False))
        self._event_value_optimizer.zero_grad(set_to_none=True)
        local_loss_sum, local_value_count = self._event_sidecar.smdp_value_loss(
            sidecar_rollout.chunk_output.representation,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            chunk_event_ids,
            gamma=float(self.cfg.algorithm.get("gamma", 1.0)),
            bootstrap_on_truncation=bool(
                self.cfg.algorithm.get("event_sidecar", {}).get("bootstrap_on_truncation", False)
            ),
            valid_mask=chunk_valid_mask,
            return_stats=True,
        )
        global_value_count = self._global_auxiliary_count(local_value_count, sidecar_device)
        global_value_sum = self._global_auxiliary_scalar(local_loss_sum)
        loss = global_value_sum / max(global_value_count, 1)
        if global_value_count and not freeze_sidecars:
            local_loss_sum.backward()
            self._all_reduce_auxiliary_gradients(
                self._event_sidecar.event_value, normalizer=global_value_count
            )
            torch.nn.utils.clip_grad_norm_(self._event_sidecar.event_value.parameters(), 1.0)
            self._event_value_optimizer.step()
            self._event_sidecar.update_target_event_value()
        return loss.detach(), influence_loss

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        return model

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        """Save π0.5 normally and persist the separate Event sidecars too."""
        super().save_checkpoint(save_path, step)
        if self._event_sidecar is None or self._rank != 0:
            return
        payload: dict[str, object] = {
            "format": "eventvalue_rl_sidecars_v2",
            "event_value": self._event_sidecar.event_value.state_dict(),
            "target_event_value": self._event_sidecar.target_event_value.state_dict(),
            "observer": self._event_sidecar.observer.state_dict(),
            "rgb_student": (
                self._event_sidecar.rgb_student.state_dict()
                if self._event_sidecar.rgb_student is not None
                else None
            ),
            "proprio_time_delta": self._event_sidecar.proprio_time_delta,
            "online_mount_token": self._event_sidecar.online_mount_token,
            "time_unit": "full_action_chunk",
            "event_value_optimizer": self._event_value_optimizer.state_dict()
            if self._event_value_optimizer is not None
            else None,
            "branch_supervision_count": self._event_branch_supervision_count,
            "branch_outcome_count": self._event_branch_outcome_count,
        }
        if self._event_influence_model is not None:
            payload["influence_model"] = self._event_influence_model.state_dict()
            payload["influence_optimizer"] = (
                self._event_influence_optimizer.state_dict()
                if self._event_influence_optimizer is not None
                else None
            )
        path = Path(save_path) / "eventvalue_sidecars.pt"
        torch.save(payload, path)

    def get_rollout_state_dict(self) -> dict:
        return self.get_model_state_dict(cpu_offload=False, full_state_dict=False)

    @Worker.timer("actor/sync_model_to_rollout")
    async def sync_model_to_rollout(self) -> None:
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        state_dict = self.get_rollout_state_dict()

        async def send_func(data):
            if not self._is_weight_sender:
                return
            await self.broadcast(
                data,
                groups=[
                    (self._group_name, 0),
                    (self._rollout_group_name, self._rollout_all_ranks),
                ],
                src=(self._group_name, 0),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        async def recv_func():
            return await self.recv(
                src_group_name=self._rollout_group_name,
                src_rank=0,
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        if not self.weight_syncer.sender_initialized():
            await self.weight_syncer.init_sender(
                state_dict=state_dict,
                send=send_func,
                recv=recv_func,
                param_names_need_sync=self.param_names_need_sync,
                is_sender=self._is_weight_sender,
            )

        version = (
            self.get_rollout_sync_version()
            if hasattr(self, "get_rollout_sync_version")
            else self.version
        )
        await self.weight_syncer.sync(state_dict, send_func, version=version)

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad(True)

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.rollout_batch = convert_trajectories_to_batch(recv_list)

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.env.train.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    @Worker.timer("actor/compute_adv")
    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """
        if self.cfg.algorithm.adv_type == "opd":
            self.compute_opd_teacher_logprobs()

        event_value_loss = None
        influence_loss = None
        # A deployable RGB student + frozen Observer produces the online event
        # state.  ``V_E`` alone is updated from the present policy rollout.
        # This must run before advantage preprocessing so values/IDs retain
        # RLinf's native [chunk,batch,action] layout.
        if self.cfg.algorithm.get("event_value_source", None) == "learned_sidecar":
            event_value_loss, influence_loss = self._populate_learned_event_sidecar()

        # Phase-2 oracle control uses the already-collected π0.5 value head as
        # a temporary bootstrap proxy.  It is deliberately explicit in config;
        # the learned Event Value Critic above replaces this field in the next
        # phase.  This branch remains intact for the oracle ablation.
        elif self.cfg.algorithm.adv_type == "event_smdp_temporal":
            if self.rollout_batch.get("event_ids") is None:
                raise RuntimeError(
                    "event_smdp_temporal needs env.train.event_oracle.enabled=true "
                    "and task oracle_event_id annotations."
                )
            if self.cfg.algorithm.get("event_value_source", None) != "pi05_value_proxy":
                raise RuntimeError(
                    "event_smdp_temporal currently supports only "
                    "algorithm.event_value_source=pi05_value_proxy."
                )
            pi05_values = self.rollout_batch.get("prev_values")
            if pi05_values is None or pi05_values.ndim != 3:
                raise RuntimeError(
                    "pi05_value_proxy expects prev_values with shape "
                    "[num_chunks + 1, batch, value_width]."
                )
            action_chunk = self.rollout_batch["event_ids"].shape[-1]
            if pi05_values.shape[-1] == 1 and action_chunk > 1:
                # π0.5's built-in critic is sampled once per policy chunk,
                # whereas Oracle Event-SMDP labels every executed action.
                # Repeat its boundary value within that chunk, retaining the
                # final boundary value required by the flattened T+1 target.
                pi05_values = pi05_values.expand(-1, -1, action_chunk).contiguous()
            elif pi05_values.shape[-1] != action_chunk:
                raise RuntimeError(
                    "pi05_value_proxy width must be 1 or match event action "
                    f"chunk width; got {pi05_values.shape[-1]} and {action_chunk}."
                )
            self.rollout_batch["event_values"] = pi05_values
            self.rollout_batch["intervention_influence"] = torch.zeros_like(
                self.rollout_batch["event_ids"], dtype=pi05_values.dtype
            )

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "prev_logprobs": self.rollout_batch.get("prev_logprobs", None),
            "teacher_logprobs": self.rollout_batch.get("teacher_logprobs", None),
            "num_action_chunks": self.cfg.actor.model.num_action_chunks,
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
            "advantage_mode": self.cfg.algorithm.get("advantage_mode", None),
            # These are produced by the optional Role-Graph Event Observer
            # sidecar.  Keeping them in the rollout batch means the π0.5
            # Flow-SDE policy and PPO loss remain untouched.
            "event_ids": self.rollout_batch.get("event_ids", None),
            "event_values": self.rollout_batch.get("event_values", None),
            "intervention_influence": self.rollout_batch.get(
                "intervention_influence", None
            ),
            "influence_temperature": self.cfg.algorithm.get(
                "influence_temperature", 1.0
            ),
            "event_mix_lambda": self._event_credit_mix_lambda(),
            "influence_beta": self.cfg.algorithm.get("event_credit", {}).get("influence_beta", 0.0),
            "influence_clip": self.cfg.algorithm.get("event_credit", {}).get("influence_clip", 3.0),
        }

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        if event_value_loss is not None:
            rollout_metrics["event/value_smdp_loss"] = event_value_loss.cpu()
        if influence_loss is not None:
            rollout_metrics["event/influence_loss"] = influence_loss.cpu()
            rollout_metrics["event/branch_supervision_count"] = torch.tensor(
                float(self._event_branch_supervision_count)
            )
            rollout_metrics["event/branch_outcome_count"] = torch.tensor(
                float(self._event_branch_outcome_count)
            )
        if self.cfg.algorithm.adv_type == "event_smdp_residual":
            rollout_metrics["event/mix_lambda"] = torch.tensor(self._event_credit_mix_lambda())

        # These tensors live on the sidecar's chunk clock.  They have already
        # served their only purposes (Value/Influence update and advantage
        # construction) and must not enter RLinf's generic PPO token shuffle:
        # the actor minibatch is action-token sized while branch/reference
        # candidates are one record per complete action chunk.  Keeping them
        # here would both waste memory and make the generic reshape/index path
        # mix incompatible time axes.
        for field_name in (
            "event_ids",
            "event_values",
            "intervention_influence",
            "branch_rewards",
            "branch_horizons",
            "branch_requested_steps",
            "branch_mask",
            "branch_valid",
            "branch_actions",
            "influence_reference_actions",
            "branch_success",
            "branch_terminations",
            "branch_truncations",
            "branch_state_ids",
            "branch_main_images",
            "branch_wrist_images",
            "branch_measured_state16",
        ):
            self.rollout_batch.pop(field_name, None)
        return rollout_metrics

    @Worker.timer("actor/compute_opd_teacher_logprobs")
    def compute_opd_teacher_logprobs(self) -> None:
        assert self.rollout_batch.get("teacher_logprobs", None) is None, (
            "OPD teacher_logprobs must be computed after rollout on actor workers."
        )
        assert self.cfg.rollout.get("expert_model", None) is not None, (
            "OPD requires rollout.expert_model as teacher model config."
        )
        assert "forward_inputs" in self.rollout_batch, (
            "OPD teacher logprob computation requires rollout forward_inputs."
        )
        assert "prev_logprobs" in self.rollout_batch, (
            "OPD teacher logprob computation requires student prev_logprobs."
        )
        assert SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ], "OPD teacher logprob computation currently supports OpenVLA models."

        prev_logprobs = self.rollout_batch["prev_logprobs"]
        time_dim, batch_dim = prev_logprobs.shape[:2]
        flat_batch_size = time_dim * batch_dim

        assert self.enable_offload and self.is_weight_offloaded, (
            "OPD teacher logprob computation expects actor weights to be "
            "offloaded before moving the teacher model to GPU."
        )
        teacher_model = self._get_opd_teacher_model()
        teacher_model.to(self.device)

        flat_forward_inputs = flatten_nested_tensor_time_batch(
            self.rollout_batch["forward_inputs"], ("forward_inputs",)
        )
        num_chunks = (
            flat_batch_size + self.cfg.actor.micro_batch_size - 1
        ) // self.cfg.actor.micro_batch_size
        teacher_logprobs = []
        kwargs = {
            "temperature": self.cfg.rollout.sampling_params.temperature_train,
            "top_k": self.cfg.rollout.sampling_params.top_k,
        }
        with torch.no_grad():
            for micro_batch in split_dict_to_chunk(flat_forward_inputs, num_chunks):
                micro_batch = put_tensor_device(micro_batch, self.device)
                with self.amp_context:
                    teacher_output = teacher_model(
                        forward_inputs=micro_batch,
                        compute_logprobs=True,
                        compute_entropy=False,
                        compute_values=False,
                        use_cache=False,
                        **kwargs,
                    )
                teacher_logprobs.append(teacher_output["logprobs"].detach().cpu())

        teacher_logprobs = torch.cat(teacher_logprobs, dim=0)
        expected_shape = (flat_batch_size, *prev_logprobs.shape[2:])
        assert teacher_logprobs.shape == expected_shape, (
            f"teacher_logprobs shape {teacher_logprobs.shape} must match "
            f"flattened student logprobs shape {expected_shape}."
        )
        self.rollout_batch["teacher_logprobs"] = teacher_logprobs.reshape(
            time_dim, batch_dim, *teacher_logprobs.shape[1:]
        )

        teacher_model.to("cpu")
        clear_memory()

    @Worker.timer("actor/recompute_logprobs")
    def recompute_prev_logprobs(self, batch_size_per_rank: int) -> dict[str, float]:
        """
        Recompute ``prev_logprobs`` with the actor's own forward, so both ends of
        the PPO ratio come from the same path. Runs after the shuffle and reuses
        the update loop's split, so each sample is scored in the micro-batch it
        will be trained in.

        Args:
            batch_size_per_rank: Samples per optimizer step on this rank.

        Returns:
            Dict with ``actor/rollout_train_logprob_gap`` (mean absolute logprob difference).
        """
        assert "forward_inputs" in self.rollout_batch, (
            "Recomputing logprobs requires rollout forward_inputs."
        )

        rollout_logprobs = self.rollout_batch["prev_logprobs"]
        rollout_size = rollout_logprobs.shape[0]
        kwargs = {
            "temperature": self.cfg.rollout.sampling_params.temperature_train,
            "top_k": self.cfg.rollout.sampling_params.top_k,
        }

        recomputed_logprobs = []
        with torch.no_grad():
            for global_batch in split_dict_to_chunk(
                self.rollout_batch, rollout_size // batch_size_per_rank
            ):
                for micro_batch in split_dict_to_chunk(
                    global_batch,
                    batch_size_per_rank // self.cfg.actor.micro_batch_size,
                ):
                    micro_batch = put_tensor_device(micro_batch, self.device)
                    with self.amp_context:
                        output = self.model(
                            forward_inputs=micro_batch["forward_inputs"],
                            compute_logprobs=True,
                            compute_entropy=False,
                            compute_values=False,
                            use_cache=False,
                            **kwargs,
                        )
                    recomputed_logprobs.append(output["logprobs"].detach().cpu())

        recomputed_logprobs = torch.cat(recomputed_logprobs, dim=0).to(
            rollout_logprobs.dtype
        )
        assert recomputed_logprobs.shape == rollout_logprobs.shape, (
            f"recomputed logprobs shape {recomputed_logprobs.shape} must match "
            f"rollout logprobs shape {rollout_logprobs.shape}."
        )
        clear_memory()

        # Logprobs are per action token, the mask and the advantages per action.
        action_dim = self.cfg.actor.model.get("action_dim", 7)
        log_ratio = (recomputed_logprobs - rollout_logprobs).reshape(
            *rollout_logprobs.shape[:-1], -1, action_dim
        )
        loss_mask = self.rollout_batch.get("loss_mask", None)
        mask = (
            None
            if loss_mask is None
            else loss_mask.bool().unsqueeze(-1).expand_as(log_ratio)
        )
        metrics = {
            "actor/rollout_train_logprob_gap": masked_mean(
                log_ratio.abs(), mask=mask
            ).item()
        }

        self.rollout_batch["prev_logprobs"] = recomputed_logprobs
        return metrics

    def _get_opd_teacher_model(self):
        if self._opd_teacher_model is None:
            teacher_model_config = build_expert_model_config(
                self.cfg, self.cfg.actor.model
            )
            teacher_model = get_model(teacher_model_config)
            if self.cfg.runner.get("expert_ckpt_path", None):
                teacher_model_dict = torch.load(
                    self.cfg.runner.expert_ckpt_path, map_location="cpu"
                )
                teacher_model.load_state_dict(teacher_model_dict)
            teacher_model.eval()
            teacher_model.requires_grad_(False)
            teacher_model.to("cpu")
            self._opd_teacher_model = teacher_model
        return self._opd_teacher_model

    def _build_sft_data_loader(self):
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.OPENPI_RLINF,
        ]:
            repo_id = resolve_lerobot_repo_id(self.cfg.actor.get("sft_data_path"))
            if repo_id is None:
                raise ValueError(
                    "actor.sft_data_path must be set to a local dataset path or "
                    "LeRobot repo id when enable_sft_co_train=True."
                )

            import openpi.training.data_loader as _data

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            training_config_name = OmegaConf.select(
                self.cfg.actor.model, "openpi.config_name", default=None
            )
            if not training_config_name:
                raise ValueError(
                    "enable_sft_co_train=True requires actor.model.openpi.config_name."
                )
            data_loader_config = get_openpi_config(
                training_config_name,
                model_path=self.cfg.actor.model.model_path,
                repo_id=repo_id,
                data_kwargs=getattr(self.cfg.actor.model, "openpi_data", None),
            )
            self.data_loader = _data.create_data_loader(
                data_loader_config, framework="pytorch", shuffle=True
            )
            self.sft_iterator = iter(self.data_loader)
            self.train_epoch = 0
            self.sft_loss_weight = self.cfg.actor.get("sft_loss_weight", 0.1)
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _train_sft_epoch(
        self, metrics_data: dict[str, torch.Tensor], loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Train one epoch of SFT.
        """
        metrics_data["ppo_loss"] = loss.clone().detach().item()

        # Get next data batch
        try:
            observation, actions = next(self.sft_iterator)
        except StopIteration:
            self.train_epoch += 1
            self.data_loader.set_epoch(self.train_epoch)
            self.sft_iterator = iter(self.data_loader)
            observation, actions = next(self.sft_iterator)

        sft_loss = self.model(
            data=(observation, actions),
            forward_type=ForwardType.SFT,
        )
        metrics_data["sft_loss"] = sft_loss.detach().item()
        total_loss = loss + self.sft_loss_weight * sft_loss
        loss = total_loss

        metrics_data["loss_ratio"] = (
            np.abs(metrics_data["sft_loss"]) / np.abs(metrics_data["ppo_loss"])
            if np.abs(metrics_data["ppo_loss"]) > 0
            else float("inf")
        )
        if metrics_data["loss_ratio"] > 1e5:
            self.logger.warning(
                "SFT/PPO loss imbalance detected: "
                f"ratio={metrics_data['loss_ratio']:.3e}, "
                f"sft_loss={metrics_data['sft_loss']:.6f}, "
                f"ppo_loss={metrics_data['ppo_loss']:.6f}, "
                f"sft_loss_weight={self.sft_loss_weight:.6f}"
            )
        return loss

    @Worker.timer("run_training")
    def run_training(self) -> None:
        """
        Run the training process using the received rollout batch.
        """
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        if self.cfg.algorithm.loss_type == "opd":
            target_steps = int(self.rollout_batch["advantages"].shape[0])
            for key in [
                "prev_logprobs",
                "forward_inputs",
                "loss_mask",
                "loss_mask_sum",
            ]:
                assert key in self.rollout_batch, f"OPD training requires {key}."
                self.rollout_batch[key] = trim_nested_tensor_time_dim(
                    self.rollout_batch[key], target_steps, (key,)
                )

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        if self.cfg.algorithm.get("recompute_logprobs", False):
            append_to_dict(metrics, self.recompute_prev_logprobs(batch_size_per_rank))
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, batch in enumerate(train_micro_batch):
                    self.train_micro_batch(
                        micro_batch=batch,
                        metrics=metrics,
                        is_last=(idx + 1) == self.gradient_accumulation,
                    )
                    # avoid gpu memory leak
                    train_micro_batch[idx] = None
                    del batch

                self.torch_platform.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        explained_variance_stats = pop_critic_explained_variance_stats(metrics)
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        if explained_variance_stats:
            reduced_stats = all_reduce_dict(
                explained_variance_stats, op=torch.distributed.ReduceOp.SUM
            )
            mean_metric_dict[CRITIC_EXPLAINED_VARIANCE_KEY] = (
                compute_critic_explained_variance_from_stats(reduced_stats).item()
            )

        return mean_metric_dict

    def train_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        metrics: dict[str, list[float]],
        *,
        is_last: bool,
    ) -> None:
        micro_batch = put_tensor_device(micro_batch, self.device)
        backward_ctx = self.before_micro_batch(self.model, is_last_micro_batch=is_last)
        advantages = micro_batch["advantages"]
        prev_logprobs = micro_batch["prev_logprobs"]
        returns = micro_batch.get("returns", None)
        prev_values = micro_batch.get("prev_values", None)
        loss_mask = micro_batch.get("loss_mask", None)
        loss_mask_sum = micro_batch.get("loss_mask_sum", None)
        forward_inputs = micro_batch.get("forward_inputs", None)

        kwargs = {}
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            kwargs["temperature"] = self.cfg.rollout.sampling_params.temperature_train
            kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            kwargs["prev_logprobs"] = prev_logprobs

        compute_values = self.cfg.algorithm.adv_type in ("gae", "event_smdp_residual")
        with self.amp_context:
            output_dict = self.model(
                forward_inputs=forward_inputs,
                compute_logprobs=True,
                compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                compute_values=compute_values,
                use_cache=False,
                **kwargs,
            )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            prev_logprobs = output_dict["prev_logprobs"]

        loss_kwargs = {
            "loss_type": self.cfg.algorithm.loss_type,
            "logprob_type": self.cfg.algorithm.logprob_type,
            "reward_type": self.cfg.algorithm.reward_type,
            "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
            "logprobs": output_dict["logprobs"],
            "values": output_dict.get("values", None),
            "old_logprobs": prev_logprobs,
            "advantages": advantages,
            "returns": returns,
            "prev_values": prev_values,
            "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
            "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
            "value_clip": self.cfg.algorithm.get("value_clip", None),
            "huber_delta": self.cfg.algorithm.get("huber_delta", None),
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "max_episode_steps": self.cfg.env.train.max_episode_steps,
            "task_type": self.cfg.runner.task_type,
            "critic_warmup": self.optimizer_steps < self.critic_warmup_steps,
        }

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
        ]:
            loss_kwargs["clip_ratio_c"] = self.cfg.algorithm.get("clip_ratio_c", 3.0)
            if self.cfg.algorithm.get("clip_log_ratio_min") is not None:
                loss_kwargs["clip_log_ratio_min"] = (
                    self.cfg.algorithm.clip_log_ratio_min
                )
            if self.cfg.algorithm.get("clip_log_ratio_max") is not None:
                loss_kwargs["clip_log_ratio_max"] = (
                    self.cfg.algorithm.clip_log_ratio_max
                )

        loss, metrics_data = policy_loss(**loss_kwargs)
        entropy_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
        if self.cfg.algorithm.entropy_bonus > 0 and not loss_kwargs["critic_warmup"]:
            entropy = output_dict["entropy"]
            entropy = reshape_entropy(
                entropy,
                entropy_type=self.cfg.algorithm.entropy_type,
                action_dim=self.cfg.actor.model.get("action_dim", 7),
                batch_size=output_dict["logprobs"].shape[0],
            )
            entropy_loss = masked_mean(entropy, mask=loss_mask)
            loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
        metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()

        if self.enable_sft_co_train:
            loss = self._train_sft_epoch(metrics_data, loss)

        loss /= self.gradient_accumulation
        with backward_ctx:
            self.grad_scaler.scale(loss).backward()

        metrics_data["actor/total_loss"] = loss.detach().item()
        append_to_dict(metrics, metrics_data)

    def set_global_step(self, global_step: int) -> None:
        """
        Set the global step for the model, if needed.
        """
        self.version = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def finish_global_batch(self, metrics: dict[str, list[float]]) -> None:
        self.torch_platform.empty_cache()
        grad_norm, lr_list = self.optimizer_step()
        self.optimizer.zero_grad()
        metric_data = {
            "actor/grad_norm": grad_norm,
            "actor/lr": lr_list[0],
        }
        if len(lr_list) > 1:
            metric_data["critic/lr"] = lr_list[1]
        append_to_dict(metrics, metric_data)
