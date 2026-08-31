# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import shutil
import uuid
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from signal_forge.hive import (
    ExplorationControllerConfig,
    HIVE_DATALOADER_CHECKPOINT_FORMAT,
    HiveComputeCounters,
    HiveEpochSpanningDataStream,
    HiveAdaptiveTopupAccumulator,
    HiveAdaptiveTopupConfig,
    HiveTopupAcquisitionDiagnostics,
    HiveTopupDataExhaustedError,
    HivePostRolloutConfig,
    HivePostRolloutInterpreter,
    HivePostRolloutResult,
    HivePreRolloutConfig,
    HivePreRolloutStep,
    HivePromptPreprocessor,
    HiveSelectorState,
    HiveSignalCounters,
    HiveSignalStepCounts,
    HiveStepPendingCommit,
    Stage1Config,
    Stage1StepSelector,
    Stage2Config,
    Stage2Selector,
    aggregate_pre_rollout_selection_metrics,
    compute_stage2_counts,
    compute_hive_group_signal_counts,
    validate_hive_prompt_preprocessing_scope,
)
from signal_forge.observability import (
    RolloutBudgetTracker,
    append_validation_reward_extra_info,
    build_validation_compute_metrics,
    compute_group_metrics,
    compute_length_metrics,
    compute_reward_extra_metrics,
    compute_section18_timing_metrics,
    compute_validation_alias_metrics,
    load_best_checkpoint_metadata,
    validate_diagnostic_validation_contract,
)

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.distillation.losses import is_distillation_enabled
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    create_rl_dataset,
    create_rl_sampler,
    need_critic,
    need_reference_policy,
    need_reward_model,
    need_teacher_policy,
)
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import deprecated, load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.skip.skip_manager import SkipManager
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import DistillationConfig, EngineConfig
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]



def _cfg_get(config, key, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _should_save_checkpoint(*, trainer_config, global_step: int, is_last_step: bool, esi: bool) -> bool:
    save_freq = int(trainer_config.get("save_freq", -1) or -1)
    raw_extra_steps = trainer_config.get("extra_save_steps", []) or []
    extra_steps = set()
    for value in raw_extra_steps:
        if isinstance(value, bool) or int(value) <= 0 or int(value) != value:
            raise ValueError("trainer.extra_save_steps must contain positive integer steps")
        extra_steps.add(int(value))
    return global_step in extra_steps or (
        save_freq > 0 and (is_last_step or global_step % save_freq == 0 or esi)
    )


def _hive_step_start_metrics(selector: Stage1StepSelector) -> dict[str, float]:
    snapshot = selector.snapshot
    return {
        "hive/selector_snapshot_global_step": float(snapshot.global_step),
        "hive/history_prompts_at_step_start": float(len(snapshot.prompt_history)),
        "hive/history_visits_at_step_start": float(
            sum(len(visits) for visits in snapshot.prompt_history.values())
        ),
        "hive/p_easy_step_start": snapshot.p_easy,
        "hive/p_hard_step_start": snapshot.p_hard,
        "hive/p_default_step_start": snapshot.p_default,
    }


def _config_to_plain_dict(config) -> dict[str, Any]:
    if OmegaConf.is_config(config):
        plain = OmegaConf.to_container(config, resolve=True)
    elif hasattr(config, "items"):
        plain = dict(config.items())
    else:
        raise TypeError(f"expected a mapping-like configuration, got {type(config).__name__}")
    if not isinstance(plain, dict):
        raise TypeError("HIVE configuration must resolve to a mapping")
    return plain


def _filter_reward_infos_by_indices(
    reward_extra_infos_dict: dict[str, list], keep_indices: np.ndarray, batch_size: int
) -> dict[str, list]:
    filtered = {}
    for key, values in reward_extra_infos_dict.items():
        if hasattr(values, "tolist"):
            values = values.tolist()
        if isinstance(values, list) and len(values) == batch_size:
            filtered[key] = np.asarray(values, dtype=object)[keep_indices].tolist()
        else:
            filtered[key] = values
    return filtered


def _concat_reward_infos(reward_info_parts: list[dict[str, list]]) -> dict[str, list]:
    merged: dict[str, list] = {}
    for reward_infos in reward_info_parts:
        for key, values in reward_infos.items():
            if hasattr(values, "tolist"):
                values = values.tolist()
            if isinstance(values, list):
                merged.setdefault(key, []).extend(values)
    return merged


def _count_prompt_groups(batch: DataProto) -> int:
    uids = np.asarray(batch.non_tensor_batch.get("uid", []), dtype=object).tolist()
    return len(set(uid for uid in uids if uid not in (None, "")))


def _refresh_training_batch_meta_info(batch: DataProto) -> None:
    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
    images_seqlens_all = []
    for multi_modal_input in batch.non_tensor_batch.get("multi_modal_inputs", []):
        if isinstance(multi_modal_input, dict) and "image_grid_thw" in multi_modal_input:
            images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
    batch.meta_info["images_seqlens"] = images_seqlens_all


def _select_first_prompt_groups(
    batch: DataProto, reward_extra_infos_dict: dict[str, list], prompt_group_count: int
) -> tuple[DataProto, dict[str, list]]:
    uids = np.asarray(batch.non_tensor_batch["uid"], dtype=object).tolist()
    selected_uids = []
    selected_uid_set = set()
    keep_indices = []
    for idx, uid in enumerate(uids):
        if uid not in selected_uid_set:
            if len(selected_uids) >= prompt_group_count:
                continue
            selected_uids.append(uid)
            selected_uid_set.add(uid)
        if uid in selected_uid_set:
            keep_indices.append(idx)

    keep_indices_np = np.asarray(keep_indices, dtype=np.int64)
    selected_batch = batch.select_idxs(keep_indices_np)
    return selected_batch, _filter_reward_infos_by_indices(reward_extra_infos_dict, keep_indices_np, len(batch))


def apply_filter_groups(
    batch: DataProto,
    reward_extra_infos_dict: dict[str, list],
    filter_groups_config,
    ppo_mini_batch_size: int,
) -> tuple[DataProto, dict[str, list], dict[str, float]]:
    """Return only mixed GRPO groups; all-correct/all-wrong groups are rejected."""
    del ppo_mini_batch_size
    if not _cfg_get(filter_groups_config, "enable", False):
        return batch, reward_extra_infos_dict, {}

    metric_name = _cfg_get(filter_groups_config, "metric", None) or "acc"
    uids = np.asarray(batch.non_tensor_batch["uid"], dtype=object)
    batch_size = len(uids)
    if metric_name in batch.non_tensor_batch:
        metric_values = np.asarray(batch.non_tensor_batch[metric_name], dtype=np.float32)
    else:
        metric_values = batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().float().numpy()

    uid_to_indices: dict[object, list[int]] = defaultdict(list)
    for idx, uid in enumerate(uids):
        uid_to_indices[uid].append(idx)

    accepted_groups = []
    all_correct_groups = 0
    all_wrong_groups = 0
    for indices in uid_to_indices.values():
        vals = metric_values[indices]
        all_correct = bool(np.all(vals > 0.0))
        all_wrong = bool(np.all(vals <= 0.0))
        all_correct_groups += int(all_correct)
        all_wrong_groups += int(all_wrong)
        if not all_correct and not all_wrong:
            accepted_groups.append(indices)

    raw_group_count = len(uid_to_indices)
    accepted_group_count = len(accepted_groups)
    rejected_group_count = raw_group_count - accepted_group_count
    keep_indices = np.asarray([idx for group in accepted_groups for idx in group], dtype=np.int64)

    metrics = {
        "dynamic_sampling/group_count": float(raw_group_count),
        "dynamic_sampling/accepted_group_count": float(accepted_group_count),
        "dynamic_sampling/rejected_group_count": float(rejected_group_count),
        "dynamic_sampling/all_correct_group_count": float(all_correct_groups),
        "dynamic_sampling/all_wrong_group_count": float(all_wrong_groups),
        "dynamic_sampling/accepted_group_ratio": float(accepted_group_count / raw_group_count) if raw_group_count else 0.0,
        "dynamic_sampling/all_correct_group_ratio": float(all_correct_groups / raw_group_count) if raw_group_count else 0.0,
        "dynamic_sampling/all_wrong_group_ratio": float(all_wrong_groups / raw_group_count) if raw_group_count else 0.0,
        "dynamic_sampling/rejected_group_ratio": float(rejected_group_count / raw_group_count) if raw_group_count else 0.0,
        "dynamic_sampling/raw_rollout_count": float(batch_size),
        "dynamic_sampling/accepted_rollout_count": float(len(keep_indices)),
        "dynamic_sampling/rejected_rollout_count": float(batch_size - len(keep_indices)),
        "dynamic_sampling/accepted_rollout_ratio": float(len(keep_indices) / batch_size) if batch_size else 0.0,
        "dynamic_sampling/raw_response_tokens": float(batch.batch["response_mask"].sum().detach().cpu().item()),
    }

    filtered_batch = batch.select_idxs(keep_indices)
    for key, value in list(filtered_batch.meta_info.items()):
        if isinstance(value, list) and len(value) == batch_size:
            filtered_batch.meta_info[key] = [value[int(i)] for i in keep_indices]
    metrics["dynamic_sampling/accepted_response_tokens"] = float(
        filtered_batch.batch["response_mask"].sum().detach().cpu().item()
    )
    return filtered_batch, _filter_reward_infos_by_indices(reward_extra_infos_dict, keep_indices, batch_size), metrics

def compute_spec_decode_metrics(
    spec_drafts,
    spec_accepts,
    spec_verifies,
    non_padding_mask=None,
) -> dict:
    """Aggregate per-request speculative decoding stats.

    Ratios are computed per request and then averaged, so long and short
    responses have equal metric weight.

    The three inputs come from the rollout engine (vLLM request spec-decode
    stats or sglang ``meta_info["spec_*"]`` keys). Either all three are ``None``
    (caller didn't fetch them, e.g. spec rollout disabled) and the function
    is a no-op, or all three are populated; mixed state is a programmer error.

    ``non_padding_mask`` is a numpy bool array used by sync PPO to drop padded
    placeholder samples; pass ``None`` for async PPO.
    """
    if spec_drafts is None and spec_accepts is None and spec_verifies is None:
        return {}
    assert spec_drafts is not None and spec_accepts is not None and spec_verifies is not None, (
        "spec_decode metrics require all three of spec_num_draft_tokens / "
        "spec_num_accepted_tokens / spec_num_verify_steps; got partial inputs"
    )

    drafts = spec_drafts.tolist() if hasattr(spec_drafts, "tolist") else list(spec_drafts)
    accepts = spec_accepts.tolist() if hasattr(spec_accepts, "tolist") else list(spec_accepts)
    verifies = spec_verifies.tolist() if hasattr(spec_verifies, "tolist") else list(spec_verifies)

    if non_padding_mask is not None:
        drafts = [d for d, keep in zip(drafts, non_padding_mask, strict=True) if keep]
        accepts = [a for a, keep in zip(accepts, non_padding_mask, strict=True) if keep]
        verifies = [v for v, keep in zip(verifies, non_padding_mask, strict=True) if keep]

    if len(drafts) == 0:
        return {}

    # Treat zero-denominator samples as 0.0 and keep them in the mean.
    per_sample_accept_rate = [(a / d) if d > 0 else 0.0 for a, d in zip(accepts, drafts, strict=True)]
    per_sample_accept_length = [(1.0 + a / v) if v > 0 else 0.0 for a, v in zip(accepts, verifies, strict=True)]

    n = len(drafts)
    return {
        "rollout/spec_accept_rate": float(sum(per_sample_accept_rate) / n),
        "rollout/spec_accept_length": float(sum(per_sample_accept_length) / n),
    }


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # old_log_probs needed for path-variance proxy: w_t = 1 - 2*exp(old_log_probs) + sum_pi_squared
            adv_kwargs["old_log_probs"] = data.batch["old_log_probs"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


@deprecated("Legacy trainer is deprecated, and wil be removed in v0.9.0. Please use `trainer.use_v1=True` instead.")
class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_teacher_policy = need_teacher_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.checkpoint_manager = None
        self._init_dump_executor()
        self._budget_tracker = RolloutBudgetTracker()
        self._best_validation_metric = None
        self._best_validation_step = None
        self._best_checkpoint_path = None

        self._hive_configuration: dict[str, Any] | None = None
        self.hive_selector_state: HiveSelectorState | None = None
        self._hive_prompt_preprocessor: HivePromptPreprocessor | None = None
        self._hive_stage2_selector: Stage2Selector | None = None
        self._hive_pre_rollout_config: HivePreRolloutConfig | None = None
        self._hive_compute_counters: HiveComputeCounters | None = None
        self._hive_signal_counters: HiveSignalCounters | None = None
        self._hive_topup_config: HiveAdaptiveTopupConfig | None = None
        self._hive_data_stream: HiveEpochSpanningDataStream | None = None
        self._restored_hive_data_stream_state: dict[str, Any] | None = None
        self._initialize_hive_selector_state()

    def _initialize_hive_selector_state(self) -> None:
        hive_config = _cfg_get(self.config.algorithm, "hive", None)
        if not _cfg_get(hive_config, "enable", False):
            return

        validate_hive_prompt_preprocessing_scope(self.config)
        if str(self.config.actor_rollout_ref.rollout.name) != "vllm":
            raise ValueError("HIVE currently requires the vLLM rollout backend")

        group_size = int(_cfg_get(hive_config, "group_size", 8))
        rollout_group_size = int(self.config.actor_rollout_ref.rollout.n)
        if group_size != 8:
            raise ValueError(f"faithful HIVE requires group_size=8; got {group_size}")
        if rollout_group_size != group_size:
            raise ValueError(
                f"HIVE group_size ({group_size}) must match actor_rollout_ref.rollout.n ({rollout_group_size})"
            )

        lambda_weight = float(_cfg_get(hive_config, "lambda_weight", 1.0))
        if lambda_weight != 1.0:
            raise ValueError("faithful HIVE requires lambda_weight=1.0")
        if self.config.trainer.total_training_steps is None:
            raise ValueError(
                "HIVE candidate accumulation consumes a variable number of raw batches; "
                "trainer.total_training_steps must be explicit"
            )
        if _cfg_get(self.config.algorithm.get("filter_groups", None), "enable", False):
            raise ValueError("HIVE must not combine with legacy filter_groups/replenish")
        if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise ValueError("HIVE reproduction requires the GRPO advantage estimator")

        effective_batch_size = int(self.config.data.train_batch_size)
        prompt_entropy_micro_batch_size = int(
            _cfg_get(hive_config, "prompt_entropy_micro_batch_size", 1)
        )
        pre_rollout_config = HivePreRolloutConfig(
            effective_batch_size=effective_batch_size,
            prompt_entropy_micro_batch_size=prompt_entropy_micro_batch_size,
        )
        stage2_config = Stage2Config(
            upper_trim_ratio=float(_cfg_get(hive_config, "upper_trim_ratio", 0.25)),
            keep_ratio=float(_cfg_get(hive_config, "keep_ratio", 0.50)),
            group_size=group_size,
        )
        raw_batch_size = int(self.config.data.get("gen_batch_size", effective_batch_size))
        if compute_stage2_counts(raw_batch_size, stage2_config).post_round_keep_count == 0:
            raise ValueError(
                "HIVE b_raw/data.gen_batch_size is too small for Stage-2 G-multiple rounding "
                f"(b_raw={raw_batch_size}, keep_ratio={stage2_config.keep_ratio}, G={group_size})"
            )

        prompt_length = int(self.config.actor_rollout_ref.rollout.prompt_length)
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {}) or {}
        self._hive_prompt_preprocessor = HivePromptPreprocessor(
            self.tokenizer,
            max_prompt_length=prompt_length,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
        )
        self._hive_stage2_selector = Stage2Selector(stage2_config)
        self._hive_pre_rollout_config = pre_rollout_config

        self._hive_topup_config = HiveAdaptiveTopupConfig(
            effective_batch_size=effective_batch_size,
            group_size=group_size,
            eta=float(_cfg_get(hive_config, "eta", 1.25)),
            b_min=int(_cfg_get(hive_config, "b_min", 64)),
            max_topup_rounds=int(_cfg_get(hive_config, "max_topup_rounds", 8)),
            survival_epsilon=float(_cfg_get(hive_config, "survival_epsilon", 1e-6)),
            controller=ExplorationControllerConfig(
                alpha_total=float(_cfg_get(hive_config, "alpha_total", 0.25)),
                delta_p=float(_cfg_get(hive_config, "delta_p", 0.01)),
                p_min=float(_cfg_get(hive_config, "p_min", 0.05)),
                p_max=float(_cfg_get(hive_config, "p_max", 0.95)),
            ),
        )

        self._hive_configuration = _config_to_plain_dict(hive_config)
        self.hive_selector_state = HiveSelectorState.create(
            group_size=group_size,
            seed=int(_cfg_get(hive_config, "seed", 42)),
            p_easy=float(_cfg_get(hive_config, "p_easy_initial", 0.5)),
            p_hard=float(_cfg_get(hive_config, "p_hard_initial", 0.5)),
            p_default=float(_cfg_get(hive_config, "p_default", 0.5)),
            configuration=self._hive_configuration,
        )
        self._hive_compute_counters = HiveComputeCounters()
        self._hive_signal_counters = HiveSignalCounters()

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    @staticmethod
    def _write_generations(inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, global_steps, max_records):
        """Write generation samples as JSONL (runs in background thread)."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{global_steps}.jsonl")

        n = len(inputs)
        if max_records is not None and max_records >= 0:
            n = min(n, int(max_records))
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, max_records=None):
        """Dump rollout/validation samples as JSONL asynchronously."""
        global_steps = self.global_steps
        future = self._dump_executor.submit(
            self._write_generations,
            inputs,
            outputs,
            gts,
            scores,
            reward_extra_infos_dict,
            dump_path,
            global_steps,
            max_records,
        )
        self._dump_futures.append(future)
        # Clean up completed futures and surface any exceptions early
        still_pending = []
        for f in self._dump_futures:
            if f.done():
                f.result()  # re-raises if the write failed
            else:
                still_pending.append(f)
        self._dump_futures = still_pending

    def _init_dump_executor(self):
        """Create or recreate the dump executor and futures list."""
        self._dump_executor = ThreadPoolExecutor(max_workers=1)
        self._dump_futures = []

    def _shutdown_dump_executor(self):
        """Drain pending dump futures and shut down the executor."""
        for f in self._dump_futures:
            f.result()
        self._dump_futures.clear()
        self._dump_executor.shutdown(wait=True)

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            max_records = self.config.trainer.get("rollout_dump_max_records", None)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = {
                k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in reward_extra_infos_dict.items()
            }
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
                max_records=max_records,
            )

    def _log_hive_round_data(
        self,
        *,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: dict,
        round_index: int,
    ) -> None:
        """Synchronously preserve HIVE reward evidence before a top-up guard can fail."""
        if self.hive_selector_state is None or not bool(
            self.config.trainer.get("hive_round_dump_enabled", False)
        ):
            return
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if not rollout_data_dir:
            return

        count = len(batch)
        extras = {
            key: (value.tolist() if hasattr(value, "tolist") else value)
            for key, value in reward_extra_infos_dict.items()
        }
        extras["hive_round_index"] = [int(round_index)] * count
        extras["response_token_count"] = (
            batch.batch["response_mask"].sum(dim=-1).detach().cpu().tolist()
        )
        for key in ("prompt_id", "data_source", "uid"):
            if key in batch.non_tensor_batch:
                value = batch.non_tensor_batch[key]
                extras[key] = value.tolist() if hasattr(value, "tolist") else list(value)

        dump_path = os.path.join(
            rollout_data_dir,
            "hive_round_diagnostics",
            f"step_{self.global_steps}",
            f"round_{round_index}",
        )
        self._write_generations(
            inputs=self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True),
            outputs=self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True),
            gts=[item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch],
            scores=reward_tensor.sum(-1).detach().cpu().tolist(),
            reward_extra_infos_dict=extras,
            dump_path=dump_path,
            global_steps=self.global_steps,
            max_records=self.config.trainer.get("rollout_dump_max_records", None),
        )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_keys = (
            {"data_source", "reward_model", "extra_info", "prompt_id", "uid"}
            & batch.non_tensor_batch.keys()
        )

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _create_hive_pre_rollout_step(
        self,
        *,
        stage1_selector: Stage1StepSelector | None = None,
        candidate_target: int | None = None,
        excluded_prompt_ids: frozenset[str] = frozenset(),
    ) -> HivePreRolloutStep:
        if (
            self.hive_selector_state is None
            or self._hive_prompt_preprocessor is None
            or self._hive_stage2_selector is None
            or self._hive_pre_rollout_config is None
        ):
            raise RuntimeError("HIVE pre-rollout components are not initialized")
        hive_config = self.config.algorithm.hive
        selector = stage1_selector or Stage1StepSelector(
            self.hive_selector_state.snapshot(),
            Stage1Config(
                lambda_weight=float(_cfg_get(hive_config, "lambda_weight", 1.0)),
                epsilon_p=float(_cfg_get(hive_config, "epsilon_p", 0.01)),
            ),
        )
        return HivePreRolloutStep(
            stage1_selector=selector,
            prompt_preprocessor=self._hive_prompt_preprocessor,
            stage2_selector=self._hive_stage2_selector,
            config=self._hive_pre_rollout_config,
            candidate_target=candidate_target,
            excluded_prompt_ids=excluded_prompt_ids,
        )

    def _select_hive_pre_rollout_candidates(
        self,
        first_batch_dict: dict[str, Any] | DataProto,
        raw_batch_iterator,
        *,
        stage1_selector: Stage1StepSelector | None = None,
        candidate_target: int | None = None,
        excluded_prompt_ids: frozenset[str] = frozenset(),
        rollout_replicas_sleeping: bool = False,
    ):
        step = self._create_hive_pre_rollout_step(
            stage1_selector=stage1_selector,
            candidate_target=candidate_target,
            excluded_prompt_ids=excluded_prompt_ids,
        )
        current_batch_dict = first_batch_dict

        while not step.is_complete:
            raw_batch = (
                current_batch_dict
                if isinstance(current_batch_dict, DataProto)
                else DataProto.from_single_dict(current_batch_dict)
            )
            raw_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            prepared = step.prepare_round(raw_batch)
            entropy_result = None
            if prepared.entropy_rpc_batch is not None:
                if not rollout_replicas_sleeping:
                    self.checkpoint_manager.sleep_replicas()
                    rollout_replicas_sleeping = True
                entropy_result = self.actor_rollout_wg.compute_prompt_entropy(prepared.entropy_rpc_batch)
            step.finish_round(prepared, entropy_result)
            if step.is_complete:
                break
            try:
                current_batch_dict = next(raw_batch_iterator)
            except StopIteration as exc:
                error_type = HiveTopupDataExhaustedError if candidate_target is not None else RuntimeError
                raise error_type(
                    "training dataloader exhausted during HIVE pre-rollout candidate accumulation: "
                    f"actual={step.candidate_actual}, target={step.candidate_target}, "
                    f"topup={candidate_target is not None}"
                ) from exc

        result = step.finalize()
        if rollout_replicas_sleeping:
            # Sleep may invalidate an IPC-loaded vLLM weight mapping after dummy initialization.
            # Reuse the canonical actor-to-rollout sync before generation instead of trusting wake alone.
            self.checkpoint_manager.update_weights(global_steps=self.global_steps)
        return result, step.stage1_selector

    def _interpret_hive_post_rollout(
        self,
        *,
        selector: Stage1StepSelector | None,
        candidate_prompt_ids: tuple[str, ...] | None,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: dict[str, list],
    ) -> HivePostRolloutResult | None:
        if self.hive_selector_state is None:
            return None
        if selector is None or candidate_prompt_ids is None:
            raise RuntimeError("HIVE post-rollout interpretation requires the step selector and candidate order")

        hive_config = self.config.algorithm.hive
        controller_config = ExplorationControllerConfig(
            alpha_total=float(_cfg_get(hive_config, "alpha_total", 0.25)),
            delta_p=float(_cfg_get(hive_config, "delta_p", 0.01)),
            p_min=float(_cfg_get(hive_config, "p_min", 0.05)),
            p_max=float(_cfg_get(hive_config, "p_max", 0.95)),
        )
        interpreter = HivePostRolloutInterpreter(
            selector_snapshot=selector.snapshot,
            config=HivePostRolloutConfig(
                effective_batch_size=int(self.config.data.train_batch_size),
                group_size=int(_cfg_get(hive_config, "group_size", 8)),
                controller=controller_config,
            ),
        )
        return interpreter.interpret(
            batch=batch,
            reward_tensor=reward_tensor,
            reward_extra_infos=reward_extra_infos_dict,
            candidate_prompt_ids=candidate_prompt_ids,
            step=self.global_steps,
        )

    def _rollout_hive_topup_candidates(
        self,
        prompt_batch: DataProto,
        *,
        curr_step_profile: bool,
    ) -> tuple[DataProto, torch.Tensor, dict[str, list], dict[str, Any]]:
        if self.hive_selector_state is None:
            raise RuntimeError("HIVE top-up rollout requires an enabled selector state")
        if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise RuntimeError("HIVE top-up rollout only supports GRPO")

        prompt_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(prompt_batch.batch))], dtype=object
        )
        gen_batch = self._get_gen_batch(prompt_batch)
        gen_batch.meta_info["global_steps"] = self.global_steps
        rollout_n = self.config.actor_rollout_ref.rollout.n
        generation_input = gen_batch.repeat(repeat_times=rollout_n, interleave=True)
        if curr_step_profile:
            self.llm_server_manager.start_profile()
        rollout_started_at = time.monotonic()
        generation_output = self.async_rollout_manager.generate_sequences(generation_input)
        self.checkpoint_manager.sleep_replicas()
        rollout_wall_seconds = time.monotonic() - rollout_started_at
        if curr_step_profile:
            self.llm_server_manager.stop_profile()

        rollout_timing = dict(generation_output.meta_info.get("timing", {}))
        rollout_timing["rollout_wall_seconds"] = rollout_wall_seconds
        generation_output.meta_info.pop("timing", None)
        batch = prompt_batch.repeat(repeat_times=rollout_n, interleave=True)
        batch = batch.union(generation_output)
        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)
        batch.meta_info["global_token_num"] = torch.sum(
            batch.batch["attention_mask"], dim=-1
        ).tolist()
        images_seqlens_all = []
        for multi_modal_input in batch.non_tensor_batch.get("multi_modal_inputs", []):
            if "image_grid_thw" in multi_modal_input:
                images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
        batch.meta_info["images_seqlens"] = images_seqlens_all
        reward_started_at = time.monotonic()
        if self.use_rm and "rm_scores" not in batch.batch:
            batch_reward = self._compute_reward_colocate(batch)
            batch = batch.union(batch_reward)
        reward_tensor, reward_extra_infos = extract_reward(batch)
        rollout_timing["reward_wall_seconds"] = time.monotonic() - reward_started_at
        return batch, reward_tensor, reward_extra_infos, rollout_timing

    def _commit_hive_step(
        self,
        selector: Stage1StepSelector | None,
        pending_commit: HiveStepPendingCommit | None,
    ) -> dict[str, float]:
        if selector is None and pending_commit is None:
            return {}
        if selector is None or pending_commit is None or self.hive_selector_state is None:
            raise RuntimeError("HIVE step commit requires selector, pending observations, and live state")
        metrics = pending_commit.commit(self.hive_selector_state, selector)
        if self._hive_compute_counters is None:
            raise RuntimeError("HIVE compute counters are not initialized")
        self._hive_compute_counters.mark_step_complete(self.global_steps)
        if self._hive_signal_counters is None:
            raise RuntimeError("HIVE signal counters are not initialized")
        self._hive_signal_counters.mark_step_complete(self.global_steps)
        if self._hive_compute_counters.global_step != self.hive_selector_state.global_step:
            raise RuntimeError("HIVE selector and compute counter steps diverged during commit")
        if self._hive_signal_counters.global_step != self.hive_selector_state.global_step:
            raise RuntimeError("HIVE selector and signal counter steps diverged during commit")
        return metrics

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _validate(self, merged: bool = False):
        validation_started_at = time.monotonic()
        validation_generated_responses = 0
        validation_prompt_tokens = 0
        validation_response_tokens = 0
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
            response_width = int(test_batch.batch["responses"].shape[-1])
            validation_generated_responses += len(test_batch)
            validation_prompt_tokens += int(
                test_batch.batch["attention_mask"][:, :-response_width].sum().detach().cpu().item()
            )
            response_mask = (
                test_batch.batch["response_mask"]
                if "response_mask" in test_batch.batch
                else compute_response_mask(test_batch)
            )
            validation_response_tokens += int(response_mask.sum().detach().cpu().item())

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            append_validation_reward_extra_info(
                reward_extra_infos_dict,
                canonical_rewards=scores,
                reward_extra_info=reward_extra_info,
            )

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
                max_records=self.config.trainer.get("validation_dump_max_records", None),
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        metrics = self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)
        validation_wall_seconds = time.monotonic() - validation_started_at
        metrics.update(
            build_validation_compute_metrics(
                generated_responses=validation_generated_responses,
                generated_prompt_tokens=validation_prompt_tokens,
                generated_response_tokens=validation_response_tokens,
                validation_n=int(self.config.actor_rollout_ref.rollout.val_kwargs.n),
                wall_time_seconds=validation_wall_seconds,
                label=self.config.trainer.get("validation_label", None),
            )
        )
        return metrics

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        metric_dict.update(compute_validation_alias_metrics(data_sources, reward_extra_infos_dict))
        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _update_best_checkpoint_metadata(self, val_metrics: dict) -> dict:
        metric_name = self.config.trainer.get("best_checkpoint_metric", "val/pass_at_1")
        metric_value = val_metrics.get(metric_name)
        if metric_value is None:
            return {}

        metric_value = float(metric_value)
        if self._best_validation_metric is not None and metric_value <= self._best_validation_metric:
            return {
                "val/best_pass_at_1": float(self._best_validation_metric),
                "val/best_step": float(self._best_validation_step if self._best_validation_step is not None else -1),
            }

        previous_best_step = self._best_validation_step
        previous_best_checkpoint_path = self._best_checkpoint_path
        self._best_validation_metric = metric_value
        self._best_validation_step = int(self.global_steps)
        checkpoint_dir = self.config.trainer.default_local_dir
        checkpoint_path = os.path.join(checkpoint_dir, f"global_step_{self.global_steps}")
        checkpoint_saved_on_update = False
        if not os.path.isdir(checkpoint_path):
            checkpoint_path = None
            if self.global_steps > 0 and self.config.trainer.get("best_checkpoint_save_on_update", False):
                print(
                    "Best validation metric improved at an unscheduled checkpoint step; "
                    f"saving full checkpoint for global_step_{self.global_steps}."
                )
                self._save_checkpoint()
                expected_checkpoint_path = os.path.join(checkpoint_dir, f"global_step_{self.global_steps}")
                if os.path.isdir(expected_checkpoint_path):
                    checkpoint_path = expected_checkpoint_path
                    checkpoint_saved_on_update = True

        deleted_previous_best_checkpoint = False
        keep_latest_unscheduled = self.config.trainer.get("best_checkpoint_keep_latest_unscheduled", False)
        if keep_latest_unscheduled and previous_best_checkpoint_path and previous_best_checkpoint_path != checkpoint_path:
            save_freq = int(self.config.trainer.get("save_freq", -1) or -1)
            previous_step = int(previous_best_step) if previous_best_step is not None else -1
            previous_is_scheduled = save_freq > 0 and previous_step > 0 and previous_step % save_freq == 0
            if not previous_is_scheduled and os.path.isdir(previous_best_checkpoint_path):
                print(
                    "Removing previous unscheduled best checkpoint "
                    f"global_step_{previous_step}: {previous_best_checkpoint_path}"
                )
                shutil.rmtree(previous_best_checkpoint_path)
                deleted_previous_best_checkpoint = True

        self._best_checkpoint_path = checkpoint_path
        metadata = {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "global_step": int(self.global_steps),
            "checkpoint_path": checkpoint_path,
            "checkpoint_saved_on_update": checkpoint_saved_on_update,
            "updated_at_unix": time.time(),
        }
        os.makedirs(checkpoint_dir, exist_ok=True)
        metadata_path = os.path.join(checkpoint_dir, "best_checkpoint.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        return {
            "val/best_pass_at_1": metric_value,
            "val/best_step": float(self.global_steps),
            "val/best_checkpoint_available": float(checkpoint_path is not None),
            "val/best_checkpoint_saved_on_update": float(checkpoint_saved_on_update),
            "val/best_previous_unscheduled_checkpoint_deleted": float(deleted_previous_best_checkpoint),
        }

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            # convert critic_cfg into TrainingWorkerConfig for the unified model engine worker
            from verl.workers.engine_workers import TrainingWorkerConfig

            orig_critic_cfg = critic_cfg
            engine_config: EngineConfig = orig_critic_cfg.engine
            engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu

            critic_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=orig_critic_cfg.model,
                engine_config=engine_config,
                optimizer_config=orig_critic_cfg.optim,
                checkpoint_config=orig_critic_cfg.checkpoint,
                extra_context=getattr(self, "_critic_extra_context", {}),
            )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/verl-project/verl/blob/master/examples/tutorial/ray/tutorial.ipynb
        # for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            # assign critic loss
            from functools import partial

            from verl.workers.utils.losses import value_loss

            value_loss_ = partial(value_loss, config=orig_critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # initialize teacher loop manager
        if self.use_teacher_policy:
            from verl.experimental.teacher_loop import MultiTeacherModelManager

            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        self.llm_server_manager = LLMServerManager.create(
            config=self.config, worker_group=self.actor_rollout_wg, rollout_resource_pool=actor_rollout_resource_pool
        )

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        # To stream teacher computation with actor rollout, we instead pass the full manager so that the
        # teacher loop workers can sleep/wake together with rollout workers
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            teacher_client=self.teacher_model_manager.get_client() if self.use_teacher_policy else None,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        # Support custom CheckpointEngineManager via config
        checkpoint_manager_class_fqn = self.config.actor_rollout_ref.rollout.get("checkpoint_manager_class")
        if checkpoint_manager_class_fqn:
            CheckpointEngineManager = load_class_from_fqn(checkpoint_manager_class_fqn, "CheckpointEngineManager")
        else:
            from verl.checkpoint_engine import CheckpointEngineManager
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        if self.hive_selector_state is not None:
            if self._hive_data_stream is None:
                raise RuntimeError("HIVE data stream is not initialized at checkpoint save")
            dataloader_checkpoint = {
                "format": HIVE_DATALOADER_CHECKPOINT_FORMAT,
                "global_step": self.global_steps,
                "dataloader_state": dataloader_state_dict,
                "hive_data_stream_state": self._hive_data_stream.state_dict(),
            }
        else:
            dataloader_checkpoint = dataloader_state_dict
        torch.save(dataloader_checkpoint, dataloader_local_path)

        if self.hive_selector_state is not None:
            if self.hive_selector_state.global_step != self.global_steps:
                raise RuntimeError(
                    "HIVE selector global_step does not match trainer checkpoint step"
                )
            if self._hive_compute_counters is None:
                raise RuntimeError("HIVE compute counters are not initialized")
            if self._hive_compute_counters.global_step != self.global_steps:
                raise RuntimeError(
                    "HIVE compute counters global_step does not match trainer checkpoint step"
                )
            if self._hive_signal_counters is None:
                raise RuntimeError("HIVE signal counters are not initialized")
            if self._hive_signal_counters.global_step != self.global_steps:
                raise RuntimeError(
                    "HIVE signal counters global_step does not match trainer checkpoint step"
                )
            self.hive_selector_state.save_checkpoint(local_global_step_folder)
            self._hive_compute_counters.save_checkpoint(local_global_step_folder)
            self._hive_signal_counters.save_checkpoint(local_global_step_folder)

        self._budget_tracker.save_checkpoint(local_global_step_folder)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        checkpoint_root = os.path.dirname(os.path.normpath(global_step_folder))
        best_metadata = load_best_checkpoint_metadata(
            checkpoint_root,
            expected_metric_name=self.config.trainer.get("best_checkpoint_metric", "val/pass_at_1"),
            resume_global_step=self.global_steps,
        )
        if best_metadata is None:
            print(
                "Warning: best_checkpoint.json is absent; best-validation state "
                "will restart at the resume boundary"
            )
        else:
            self._best_validation_metric = best_metadata.metric_value
            self._best_validation_step = best_metadata.global_step
            self._best_checkpoint_path = best_metadata.checkpoint_path
            print(
                "Restored best validation metadata: "
                f"step={best_metadata.global_step}, metric={best_metadata.metric_name}, "
                f"value={best_metadata.metric_value}, checkpoint={best_metadata.checkpoint_path}"
            )

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_checkpoint = torch.load(dataloader_local_path, weights_only=False)
            is_hive_envelope = (
                isinstance(dataloader_checkpoint, dict)
                and dataloader_checkpoint.get("format") == HIVE_DATALOADER_CHECKPOINT_FORMAT
            )
            if is_hive_envelope:
                stream_global_step = dataloader_checkpoint.get("global_step")
                if stream_global_step != self.global_steps:
                    raise ValueError(
                        f"HIVE dataloader global_step {stream_global_step!r} does not match "
                        f"trainer checkpoint step {self.global_steps}"
                    )
                if self.hive_selector_state is None:
                    raise ValueError("cannot restore a HIVE dataloader checkpoint with HIVE disabled")
                self.train_dataloader.load_state_dict(dataloader_checkpoint["dataloader_state"])
                stream_state = dataloader_checkpoint.get("hive_data_stream_state")
                if not isinstance(stream_state, dict):
                    raise ValueError("HIVE dataloader checkpoint is missing data stream state")
                self._restored_hive_data_stream_state = stream_state
            elif self.hive_selector_state is not None:
                print("Warning: restoring legacy HIVE checkpoint without epoch-spanning stream state")
                self.train_dataloader.load_state_dict(dataloader_checkpoint)
            else:
                steps_per_epoch = len(self.train_dataloader)
                at_epoch_boundary = steps_per_epoch > 0 and self.global_steps % steps_per_epoch == 0
                if at_epoch_boundary:
                    print(
                        f"Skipping dataloader state restore: global_steps={self.global_steps} "
                        f"is at an epoch boundary (steps_per_epoch={steps_per_epoch}). "
                        f"The saved state marks the dataloader as exhausted. "
                        f"Next epoch will iterate from scratch."
                    )
                else:
                    self.train_dataloader.load_state_dict(dataloader_checkpoint)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        budget_checkpoint_missing = False
        try:
            self._budget_tracker = RolloutBudgetTracker.load_checkpoint(
                global_step_folder, expected_optimizer_steps=self.global_steps
            )
        except FileNotFoundError:
            budget_checkpoint_missing = True
            print(
                "Warning: rollout budget counters are absent from this checkpoint; "
                "recovering the counters available from legacy state"
            )
            self._budget_tracker = RolloutBudgetTracker(optimizer_steps=self.global_steps)

        if self.hive_selector_state is not None:
            restored_hive_state = HiveSelectorState.load_checkpoint(global_step_folder)
            if restored_hive_state.global_step != self.global_steps:
                raise ValueError(
                    f"HIVE selector global_step {restored_hive_state.global_step} does not match "
                    f"trainer checkpoint step {self.global_steps}"
                )
            if restored_hive_state.configuration != self._hive_configuration:
                raise ValueError("HIVE configuration does not match the selector checkpoint")
            self.hive_selector_state = restored_hive_state
            try:
                self._hive_compute_counters = HiveComputeCounters.load_checkpoint(
                    global_step_folder, expected_global_step=self.global_steps
                )
            except FileNotFoundError:
                print("Warning: HIVE compute counters are absent from this checkpoint; resetting them to zero")
                self._hive_compute_counters = HiveComputeCounters(global_step=self.global_steps)

            try:
                self._hive_signal_counters = HiveSignalCounters.load_checkpoint(
                    global_step_folder, expected_global_step=self.global_steps
                )
            except FileNotFoundError:
                if self.config.trainer.get("require_hive_signal_counters", False):
                    raise RuntimeError(
                        "HIVE signal counters are absent from the resume checkpoint; "
                        "run the approved observability backfill before resuming"
                    )
                print(
                    "Warning: HIVE signal counters are absent from this checkpoint; "
                    "exact signal observations start at the resume boundary"
                )
                self._hive_signal_counters = HiveSignalCounters(
                    global_step=self.global_steps,
                    candidate_observation_start_step=self.global_steps,
                    training_observation_start_step=self.global_steps,
                )

            if budget_checkpoint_missing:
                hive_counters = self._hive_compute_counters
                self._budget_tracker.candidate_prompt_groups = hive_counters.generated_prompt_groups
                self._budget_tracker.accepted_prompt_groups = hive_counters.effective_prompt_groups
                self._budget_tracker.rejected_prompt_groups = (
                    hive_counters.generated_prompt_groups - hive_counters.effective_prompt_groups
                )
                self._budget_tracker.responses_generated = hive_counters.generated_responses
                self._budget_tracker.response_tokens_generated = hive_counters.generated_response_tokens
                self._budget_tracker.rollout_tokens_generated = hive_counters.generated_response_tokens
                self._budget_tracker.effective_prompt_groups = hive_counters.effective_prompt_groups
                self._budget_tracker.effective_responses = hive_counters.effective_responses
                self._budget_tracker.effective_response_tokens = hive_counters.effective_response_tokens
                print(
                    "Warning: legacy HIVE checkpoint restored shared generated/effective compute counters; "
                    "historical prompt-token and wall-clock counters restart at the resume boundary"
                )


    def _initialize_hive_data_stream_for_fit(self) -> None:
        if self.hive_selector_state is None:
            return
        raw_batch_size = int(
            self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        )
        self._hive_data_stream = HiveEpochSpanningDataStream(
            self.train_dataloader,
            total_epochs=int(self.config.trainer.total_epochs),
            raw_batch_size=raw_batch_size,
            state=self._restored_hive_data_stream_state,
        )
        self._restored_hive_data_stream_state = None

    def _iter_hive_optimizer_steps(self):
        if self._hive_data_stream is None:
            raise RuntimeError("HIVE epoch-spanning data stream is not initialized")
        while True:
            raw_batch_iterator = self._hive_data_stream.begin_step()
            try:
                first_batch = next(raw_batch_iterator)
            except StopIteration as exc:
                raise HiveTopupDataExhaustedError(
                    "HIVE raw prompt stream exhausted before total_training_steps was reached: "
                    f"global_step={self.global_steps}, total_training_steps={self.total_training_steps}, "
                    f"epoch={self._hive_data_stream.epoch_index}, "
                    f"total_epochs={self._hive_data_stream.total_epochs}"
                ) from exc
            yield self._hive_data_stream.epoch_index, first_batch, raw_batch_iterator

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        tu.assign_non_tensor(batch_td, compute_loss=False)
        output = self.critic_wg.infer_batch(batch_td)
        output = output.get()
        values = tu.get(output, "values")
        values = no_padding_2_padding(values, batch_td)
        values = tu.get_tensordict({"values": values.float()})
        values = DataProto.from_tensordict(values)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        metadata = {"calculate_entropy": False, "compute_loss": False}
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        tu.assign_non_tensor(batch_td, **metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
        # gather output
        log_probs = tu.get(output, "log_probs")
        # step 4. No padding to padding
        log_probs = no_padding_2_padding(log_probs, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
        ref_log_prob = DataProto.from_tensordict(ref_log_prob)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        calculate_sum_pi_squared = self.config.actor_rollout_ref.actor.get("calculate_sum_pi_squared", False)
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=True,
            calculate_sum_pi_squared=calculate_sum_pi_squared,
            compute_loss=False,
        )
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        # gather output
        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        routed_experts = tu.get(output, "routed_experts")
        sum_pi_squared = tu.get(output, "sum_pi_squared") if calculate_sum_pi_squared else None

        old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
        # step 4. No padding to padding
        entropy = no_padding_2_padding(entropy, batch_td)
        log_probs = no_padding_2_padding(log_probs, batch_td)
        if sum_pi_squared is not None:
            sum_pi_squared = no_padding_2_padding(sum_pi_squared, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        result = {"old_log_probs": log_probs.float(), "entropys": entropy.float()}
        if routed_experts is not None:
            result["routed_experts"] = routed_experts
        if sum_pi_squared is not None:
            result["sum_pi_squared"] = sum_pi_squared.float()
        old_log_prob = tu.get_tensordict(result)
        old_log_prob = DataProto.from_tensordict(old_log_prob)
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        distillation_only = False  # distillation_only flag means we can skip policy loss and reduce mem footprint
        if is_distillation_enabled(self.config.get("distillation")):
            distillation_loss_cfg = self.distillation_config.distillation_loss
            distillation_only = (
                distillation_use_topk
                and not distillation_loss_cfg.use_task_rewards
                and not distillation_loss_cfg.use_policy_gradient
            )
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            distillation_use_topk=distillation_use_topk,
            distillation_only=distillation_only,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
            compute_loss=True,
        )
        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        # modify key name
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        ppo_epochs = self.config.critic.ppo_epochs
        seed = self.config.critic.data_loader_seed
        shuffle = self.config.critic.shuffle
        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )

        output = self.critic_wg.train_mini_batch(batch_td)
        output = output.get()
        output = tu.get(output, "metrics")
        output = rename_dict(output, "critic/")
        # modify key name
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        if self._dump_executor._shutdown:
            self._init_dump_executor()

        validate_diagnostic_validation_contract(self.config)

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        if self.hive_selector_state is None:
            current_epoch = self.global_steps // len(self.train_dataloader)
        else:
            self._initialize_hive_data_stream_for_fit()
            if self._hive_data_stream is None:
                raise RuntimeError("HIVE data stream initialization failed")
            current_epoch = self._hive_data_stream.epoch_index

        SkipManager.init(self.config)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            if self.config.trainer.get("update_best_checkpoint_metadata", True):
                val_metrics.update(self._update_best_checkpoint_metadata(val_metrics))
            val_metrics["training/global_step"] = float(self.global_steps)
            val_metrics.update(
                self._budget_tracker.snapshot(n_gpus=self.resource_pool_manager.get_n_gpus())
            )
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        SkipManager.set_step(self.global_steps)

        dynamic_filter_state: dict[str, Any] | None = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        epoch_sources = (
            (None,)
            if self.hive_selector_state is not None
            else range(current_epoch, self.config.trainer.total_epochs)
        )
        for epoch_source in epoch_sources:
            if epoch_source is None:
                step_sources = self._iter_hive_optimizer_steps()
            else:
                epoch = int(epoch_source)
                raw_batch_iterator = iter(self.train_dataloader)
                step_sources = (
                    (epoch, batch_dict, raw_batch_iterator) for batch_dict in raw_batch_iterator
                )
            for epoch, batch_dict, raw_batch_iterator in step_sources:
                iteration_started_at = time.monotonic()
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}
                hive_stage1_selector = None
                hive_candidate_prompt_ids = None
                hive_pending_commit = None
                hive_post_result = None
                hive_generated_reward_infos = None
                hive_generated_uids = None
                hive_generated_scalar_rewards = None
                hive_generated_raw_correctness = None
                hive_generated_response_lengths = None
                hive_pre_rollout_results = []

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                if self.hive_selector_state is None:
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                else:
                    hive_result, hive_stage1_selector = self._select_hive_pre_rollout_candidates(
                        batch_dict,
                        raw_batch_iterator,
                    )
                    batch = hive_result.selected_batch
                    metrics.update(hive_result.metrics)
                    metrics.update(_hive_step_start_metrics(hive_stage1_selector))
                    hive_candidate_prompt_ids = tuple(
                        np.asarray(batch.non_tensor_batch["prompt_id"], dtype=object).tolist()
                    )
                    hive_pre_rollout_results.append(hive_result)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                rollout_n = self.config.actor_rollout_ref.rollout.n
                gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    # NOTE: REMAX needs one sampled rollout plus one greedy baseline per prompt.
                    # Keep them in a single agent-loop/vLLM request to avoid sending a second
                    # rollout after replicas have been put to sleep, which can leave async vLLM
                    # engines in an invalid state for multi-turn agent workloads.
                    gen_batch_output.non_tensor_batch["__do_sample__"] = np.ones(len(gen_batch_output), dtype=bool)
                    gen_baseline_batch = gen_batch.slice(0, None)
                    gen_baseline_batch.non_tensor_batch["__do_sample__"] = np.zeros(len(gen_baseline_batch), dtype=bool)
                    combined_gen_batch = DataProto.concat([gen_batch_output, gen_baseline_batch])
                    num_sampled_prompts = len(gen_batch_output)
                else:
                    combined_gen_batch = gen_batch_output
                    num_sampled_prompts = len(gen_batch_output)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.llm_server_manager.start_profile()
                        combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.llm_server_manager.stop_profile()

                        timing_raw.update(combined_gen_output.meta_info["timing"])
                        combined_gen_output.meta_info.pop("timing", None)

                    gen_batch_output = combined_gen_output.slice(0, num_sampled_prompts)
                    if "__do_sample__" in gen_batch_output.non_tensor_batch:
                        gen_batch_output.pop(non_tensor_batch_keys=["__do_sample__"])

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        gen_baseline_output = combined_gen_output.slice(num_sampled_prompts, None)
                        if "__do_sample__" in gen_baseline_output.non_tensor_batch:
                            gen_baseline_output.pop(non_tensor_batch_keys=["__do_sample__"])

                        if self.use_rm and "rm_scores" not in gen_baseline_output.batch.keys():
                            baseline_reward = self._compute_reward_colocate(gen_baseline_output)
                            gen_baseline_output = gen_baseline_output.union(baseline_reward)

                        reward_baseline_tensor = gen_baseline_output.batch["rm_scores"].sum(dim=-1)
                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_output
                    del combined_gen_batch, combined_gen_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch and self.hive_selector_state is None:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    self._log_hive_round_data(
                        batch=batch,
                        reward_tensor=reward_tensor,
                        reward_extra_infos_dict=reward_extra_infos_dict,
                        round_index=0,
                    )

                    hive_post_result = self._interpret_hive_post_rollout(
                        selector=hive_stage1_selector,
                        candidate_prompt_ids=hive_candidate_prompt_ids,
                        batch=batch,
                        reward_tensor=reward_tensor,
                        reward_extra_infos_dict=reward_extra_infos_dict,
                    )
                    if hive_post_result is not None:
                        if self._hive_topup_config is None or hive_stage1_selector is None:
                            raise RuntimeError("HIVE adaptive top-up components are not initialized")
                        accumulator = HiveAdaptiveTopupAccumulator(
                            selector_snapshot=hive_stage1_selector.snapshot,
                            config=self._hive_topup_config,
                        )
                        accumulator.observe_initial(
                            hive_post_result,
                            HiveTopupAcquisitionDiagnostics(
                                candidate_target=int(hive_result.metrics["hive/candidate_target"]),
                                candidate_actual=int(hive_result.metrics["hive/candidate_actual"]),
                                raw_prompts_seen=int(hive_result.metrics["hive/raw_prompts_seen"]),
                            ),
                        )
                        if self._hive_compute_counters is None:
                            raise RuntimeError("HIVE compute counters are not initialized")
                        compute_counter_metrics = self._hive_compute_counters.update(
                            hive_post_result.diagnostics
                        )
                        generated_reward_parts = [reward_extra_infos_dict]
                        generated_uid_values = np.asarray(
                            batch.non_tensor_batch["uid"], dtype=object
                        ).tolist()
                        generated_scalar_rewards = np.asarray(
                            reward_extra_infos_dict["reward"], dtype=object
                        ).tolist()
                        generated_raw_correctness = np.asarray(
                            reward_extra_infos_dict.get(
                                "raw_correctness", reward_extra_infos_dict.get("acc", [])
                            ),
                            dtype=object,
                        ).tolist()
                        generated_response_lengths = (
                            batch.batch["response_mask"].sum(dim=-1).detach().cpu().float().numpy().tolist()
                        )

                        with marked_timer("topup", timing_raw, color="yellow"):
                            while not accumulator.is_complete:
                                estimate = accumulator.plan_next_topup()
                                if estimate is None:
                                    break
                                try:
                                    first_topup_batch = next(raw_batch_iterator)
                                except StopIteration as exc:
                                    raise HiveTopupDataExhaustedError(
                                        "training dataloader exhausted before HIVE filled the effective batch: "
                                        f"effective={accumulator.effective_group_count}, "
                                        f"required={self._hive_topup_config.effective_batch_size}, "
                                        f"topup_rounds={accumulator.topup_rounds}",
                                        diagnostics=accumulator.failure_diagnostics(),
                                    ) from exc
                                try:
                                    topup_pre_result, topup_selector = (
                                        self._select_hive_pre_rollout_candidates(
                                            first_topup_batch,
                                            raw_batch_iterator,
                                            stage1_selector=hive_stage1_selector,
                                            candidate_target=estimate.candidate_target,
                                            excluded_prompt_ids=accumulator.prompt_ids,
                                            rollout_replicas_sleeping=True,
                                        )
                                    )
                                except HiveTopupDataExhaustedError as exc:
                                    raise HiveTopupDataExhaustedError(
                                        str(exc),
                                        diagnostics=accumulator.failure_diagnostics(),
                                    ) from exc
                                if topup_selector is not hive_stage1_selector:
                                    raise RuntimeError("HIVE top-up replaced the frozen step selector")
                                hive_pre_rollout_results.append(topup_pre_result)
                                topup_prompt_ids = tuple(
                                    np.asarray(
                                        topup_pre_result.selected_batch.non_tensor_batch["prompt_id"],
                                        dtype=object,
                                    ).tolist()
                                )
                                topup_batch, topup_reward_tensor, topup_reward_infos, topup_timing = (
                                    self._rollout_hive_topup_candidates(
                                        topup_pre_result.selected_batch,
                                        curr_step_profile=curr_step_profile,
                                    )
                                )
                                self._log_hive_round_data(
                                    batch=topup_batch,
                                    reward_tensor=topup_reward_tensor,
                                    reward_extra_infos_dict=topup_reward_infos,
                                    round_index=accumulator.topup_rounds + 1,
                                )
                                topup_post_result = self._interpret_hive_post_rollout(
                                    selector=hive_stage1_selector,
                                    candidate_prompt_ids=topup_prompt_ids,
                                    batch=topup_batch,
                                    reward_tensor=topup_reward_tensor,
                                    reward_extra_infos_dict=topup_reward_infos,
                                )
                                if topup_post_result is None:
                                    raise RuntimeError("HIVE top-up interpretation unexpectedly bypassed")
                                accumulator.observe_topup(
                                    topup_post_result,
                                    HiveTopupAcquisitionDiagnostics(
                                        candidate_target=estimate.candidate_target,
                                        candidate_actual=len(topup_pre_result.selected_batch),
                                        raw_prompts_seen=int(
                                            topup_pre_result.metrics["hive/raw_prompts_seen"]
                                        ),
                                    ),
                                )
                                compute_counter_metrics = self._hive_compute_counters.update(
                                    topup_post_result.diagnostics
                                )
                                generated_reward_parts.append(topup_reward_infos)
                                generated_uid_values.extend(
                                    np.asarray(
                                        topup_batch.non_tensor_batch["uid"], dtype=object
                                    ).tolist()
                                )
                                generated_scalar_rewards.extend(
                                    np.asarray(topup_reward_infos["reward"], dtype=object).tolist()
                                )
                                generated_raw_correctness.extend(
                                    np.asarray(
                                        topup_reward_infos.get(
                                            "raw_correctness", topup_reward_infos.get("acc", [])
                                        ),
                                        dtype=object,
                                    ).tolist()
                                )
                                generated_response_lengths.extend(
                                    topup_batch.batch["response_mask"]
                                    .sum(dim=-1)
                                    .detach()
                                    .cpu()
                                    .float()
                                    .numpy()
                                    .tolist()
                                )
                                round_index = accumulator.topup_rounds
                                metrics[f"hive/topup_round_{round_index}/raw_prompts_seen"] = float(
                                    topup_pre_result.metrics["hive/raw_prompts_seen"]
                                )
                                metrics[f"hive/topup_round_{round_index}/stage2_kept"] = float(
                                    topup_pre_result.metrics["hive/stage2_kept"]
                                )
                                stage1_latency = float(
                                    topup_pre_result.metrics["hive/stage1_latency_seconds"]
                                )
                                metrics[f"hive/topup_round_{round_index}/stage1_latency_seconds"] = (
                                    stage1_latency
                                )
                                metrics["hive/stage1_latency_seconds"] = float(
                                    metrics.get("hive/stage1_latency_seconds", 0.0)
                                ) + stage1_latency
                                entropy_latency = float(
                                    topup_pre_result.metrics["hive/stage2_entropy_latency_seconds"]
                                )
                                metrics[
                                    f"hive/topup_round_{round_index}/stage2_entropy_latency_seconds"
                                ] = entropy_latency
                                metrics["hive/stage2_entropy_latency_seconds"] = float(
                                    metrics.get("hive/stage2_entropy_latency_seconds", 0.0)
                                ) + entropy_latency
                                for entropy_peak_key in (
                                    "hive/stage2_entropy_peak_allocated_bytes",
                                    "hive/stage2_entropy_peak_reserved_bytes",
                                ):
                                    round_peak = float(topup_pre_result.metrics[entropy_peak_key])
                                    metrics[f"hive/topup_round_{round_index}/{entropy_peak_key[5:]}"] = round_peak
                                    metrics[entropy_peak_key] = max(
                                        float(metrics.get(entropy_peak_key, 0.0)), round_peak
                                    )
                                for key, value in topup_timing.items():
                                    timing_key = f"topup/{key}"
                                    timing_raw[timing_key] = timing_raw.get(timing_key, 0.0) + float(value)

                        hive_final_result = accumulator.finalize(step=self.global_steps)
                        hive_post_result = hive_final_result
                        hive_generated_reward_infos = _concat_reward_infos(generated_reward_parts)
                        hive_generated_uids = generated_uid_values
                        hive_generated_scalar_rewards = generated_scalar_rewards
                        hive_generated_raw_correctness = generated_raw_correctness
                        hive_generated_response_lengths = generated_response_lengths
                        metrics.update(hive_final_result.metrics)
                        metrics.update(
                            aggregate_pre_rollout_selection_metrics(hive_pre_rollout_results)
                        )
                        metrics.update(compute_counter_metrics)
                        diagnostics = hive_final_result.diagnostics
                        metrics.update(
                            self._budget_tracker.update(
                                candidate_prompt_groups_step=diagnostics.generated_prompt_groups,
                                accepted_prompt_groups_step=diagnostics.training_prompt_groups,
                                responses_generated_step=diagnostics.generated_responses,
                                prompt_tokens_generated_step=diagnostics.generated_prompt_tokens,
                                response_tokens_generated_step=diagnostics.generated_response_tokens,
                                effective_prompt_groups_step=diagnostics.effective_prompt_groups,
                                effective_responses_step=diagnostics.effective_responses,
                                effective_response_tokens_step=diagnostics.effective_response_tokens,
                                rollout_time_seconds_step=float(
                                    timing_raw.get("gen", 0.0)
                                    + timing_raw.get("topup/rollout_wall_seconds", 0.0)
                                ),
                                optimizer_steps_step=1,
                                n_gpus=self.resource_pool_manager.get_n_gpus(),
                            )
                        )
                        batch = hive_final_result.training_batch
                        reward_tensor = hive_final_result.training_reward_tensor
                        reward_extra_infos_dict = hive_final_result.training_reward_extra_infos
                        hive_pending_commit = hive_final_result.pending_commit
                        if self._hive_signal_counters is None:
                            raise RuntimeError("HIVE signal counters are not initialized")
                        group_size = int(self.config.algorithm.hive.group_size)
                        candidate_signal_counts = compute_hive_group_signal_counts(
                            uids=hive_generated_uids,
                            scalar_rewards=hive_generated_scalar_rewards,
                            raw_correctness=hive_generated_raw_correctness,
                            group_size=group_size,
                        )
                        training_raw_correctness = reward_extra_infos_dict.get(
                            "raw_correctness", reward_extra_infos_dict.get("acc", [])
                        )
                        training_signal_counts = compute_hive_group_signal_counts(
                            uids=np.asarray(batch.non_tensor_batch["uid"], dtype=object).tolist(),
                            scalar_rewards=reward_extra_infos_dict["reward"],
                            raw_correctness=training_raw_correctness,
                            group_size=group_size,
                        )
                        metrics.update(
                            self._hive_signal_counters.update(
                                HiveSignalStepCounts(
                                    candidate=candidate_signal_counts,
                                    training=training_signal_counts,
                                    generated_response_tokens=diagnostics.generated_response_tokens,
                                    topup_groups=int(metrics["hive/generated_groups_topup"]),
                                )
                            )
                        )
                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)
                        _refresh_training_batch_meta_info(batch)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'
                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if hive_post_result is None:
                            candidate_batch_size = len(batch)
                            candidate_uids = batch.non_tensor_batch.get("uid", [])
                            candidate_uid_values = np.asarray(candidate_uids, dtype=object).tolist()
                            candidate_group_count = len(
                                set(uid for uid in candidate_uid_values if uid not in (None, ""))
                            )
                            candidate_response_lengths = (
                                batch.batch["response_mask"].sum(dim=-1).detach().cpu().float().numpy().tolist()
                            )
                            response_width = batch.batch["responses"].shape[-1]
                            candidate_prompt_tokens = int(
                                batch.batch["attention_mask"][:, :-response_width].sum().detach().cpu().item()
                            )
                            candidate_response_tokens = int(batch.batch["response_mask"].sum().detach().cpu().item())
                            raw_correctness_values = reward_extra_infos_dict.get(
                                "raw_correctness", reward_extra_infos_dict.get("acc", [])
                            )
                            candidate_reward_extra_infos_dict = reward_extra_infos_dict
                        else:
                            diagnostics = hive_post_result.diagnostics
                            candidate_batch_size = diagnostics.generated_responses
                            candidate_uids = hive_generated_uids
                            candidate_uid_values = hive_generated_uids
                            candidate_group_count = diagnostics.generated_prompt_groups
                            candidate_response_lengths = hive_generated_response_lengths
                            candidate_prompt_tokens = diagnostics.generated_prompt_tokens
                            candidate_response_tokens = diagnostics.generated_response_tokens
                            raw_correctness_values = hive_generated_raw_correctness
                            candidate_reward_extra_infos_dict = hive_generated_reward_infos
                        filter_groups_config = self.config.algorithm.get("filter_groups", None)
                        dynamic_filter_enabled = _cfg_get(filter_groups_config, "enable", False)

                        batch, reward_extra_infos_dict, filter_group_metrics = apply_filter_groups(
                            batch=batch,
                            reward_extra_infos_dict=reward_extra_infos_dict,
                            filter_groups_config=filter_groups_config,
                            ppo_mini_batch_size=self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
                        )

                        if dynamic_filter_enabled:
                            if dynamic_filter_state is None:
                                dynamic_filter_state = {
                                    "accepted_batches": [],
                                    "accepted_reward_infos": [],
                                    "candidate_reward_infos": [],
                                    "candidate_uids": [],
                                    "candidate_raw_correctness": [],
                                    "candidate_response_lengths": [],
                                    "candidate_group_count": 0,
                                    "accepted_group_count_total": 0,
                                    "all_correct_group_count": 0,
                                    "all_wrong_group_count": 0,
                                    "responses_generated": 0,
                                    "prompt_tokens_generated": 0,
                                    "response_tokens_generated": 0,
                                    "num_gen_batches": 0,
                                }

                            dynamic_filter_state["num_gen_batches"] += 1
                            dynamic_filter_state["candidate_group_count"] += candidate_group_count
                            dynamic_filter_state["accepted_group_count_total"] += int(
                                filter_group_metrics.get("dynamic_sampling/accepted_group_count", 0.0)
                            )
                            dynamic_filter_state["all_correct_group_count"] += int(
                                filter_group_metrics.get("dynamic_sampling/all_correct_group_count", 0.0)
                            )
                            dynamic_filter_state["all_wrong_group_count"] += int(
                                filter_group_metrics.get("dynamic_sampling/all_wrong_group_count", 0.0)
                            )
                            dynamic_filter_state["responses_generated"] += candidate_batch_size
                            dynamic_filter_state["prompt_tokens_generated"] += candidate_prompt_tokens
                            dynamic_filter_state["response_tokens_generated"] += candidate_response_tokens
                            dynamic_filter_state["candidate_reward_infos"].append(candidate_reward_extra_infos_dict)
                            dynamic_filter_state["candidate_uids"].extend(candidate_uid_values)
                            dynamic_filter_state["candidate_raw_correctness"].extend(
                                np.asarray(raw_correctness_values, dtype=object).tolist()
                            )
                            dynamic_filter_state["candidate_response_lengths"].extend(candidate_response_lengths)

                            accepted_groups_this_batch = _count_prompt_groups(batch)
                            if accepted_groups_this_batch > 0:
                                dynamic_filter_state["accepted_batches"].append(batch)
                                dynamic_filter_state["accepted_reward_infos"].append(reward_extra_infos_dict)

                            num_prompt_in_batch = sum(
                                _count_prompt_groups(part) for part in dynamic_filter_state["accepted_batches"]
                            )
                            prompt_bsz = int(self.config.data.train_batch_size)
                            max_num_gen_batches = int(_cfg_get(filter_groups_config, "max_num_gen_batches", 0) or 0)
                            if num_prompt_in_batch < prompt_bsz:
                                if max_num_gen_batches <= 0 or dynamic_filter_state["num_gen_batches"] < max_num_gen_batches:
                                    print(
                                        f"Dynamic sampling accepted {num_prompt_in_batch}/{prompt_bsz} prompt groups "
                                        f"after {dynamic_filter_state['num_gen_batches']} generation batch(es); keep generating."
                                    )
                                    # Generation put rollout replicas to sleep; wake them before the next candidate batch.
                                    self.checkpoint_manager.update_weights(self.global_steps)
                                    continue
                                raise ValueError(
                                    f"Dynamic sampling accepted only {num_prompt_in_batch}/{prompt_bsz} prompt groups "
                                    f"after max_num_gen_batches={max_num_gen_batches}. Check data/model/reward signal."
                                )

                            for accepted_part in dynamic_filter_state["accepted_batches"]:
                                accepted_part.meta_info.pop("global_token_num", None)
                                accepted_part.meta_info.pop("images_seqlens", None)
                            batch = DataProto.concat(dynamic_filter_state["accepted_batches"])
                            _refresh_training_batch_meta_info(batch)
                            reward_extra_infos_dict = _concat_reward_infos(dynamic_filter_state["accepted_reward_infos"])
                            batch, reward_extra_infos_dict = _select_first_prompt_groups(
                                batch, reward_extra_infos_dict, prompt_bsz
                            )
                            _refresh_training_batch_meta_info(batch)
                            accepted_group_count = _count_prompt_groups(batch)
                            candidate_reward_extra_infos = _concat_reward_infos(
                                dynamic_filter_state["candidate_reward_infos"]
                            )
                            metrics.update(compute_reward_extra_metrics(candidate_reward_extra_infos))
                            metrics.update(
                                compute_group_metrics(
                                    uids=dynamic_filter_state["candidate_uids"],
                                    raw_correctness=dynamic_filter_state["candidate_raw_correctness"],
                                    expected_group_size=self.config.actor_rollout_ref.rollout.n,
                                )
                            )
                            metrics.update(
                                compute_length_metrics(
                                    response_lengths=dynamic_filter_state["candidate_response_lengths"],
                                    raw_correctness=dynamic_filter_state["candidate_raw_correctness"],
                                    max_response_length=batch.batch["responses"].shape[-1],
                                )
                            )
                            generated_groups = int(dynamic_filter_state["candidate_group_count"])
                            rejected_groups = max(generated_groups - accepted_group_count, 0)
                            generated_rollouts = int(dynamic_filter_state["responses_generated"])
                            accepted_rollouts = len(batch)
                            generated_response_tokens = int(dynamic_filter_state["response_tokens_generated"])
                            accepted_response_tokens = int(batch.batch["response_mask"].sum().detach().cpu().item())
                            metrics.update(
                                {
                                    "dynamic_sampling/num_gen_batches": float(dynamic_filter_state["num_gen_batches"]),
                                    "dynamic_sampling/group_count": float(generated_groups),
                                    "dynamic_sampling/accepted_group_count": float(accepted_group_count),
                                    "dynamic_sampling/rejected_group_count": float(rejected_groups),
                                    "dynamic_sampling/all_correct_group_count": float(
                                        dynamic_filter_state["all_correct_group_count"]
                                    ),
                                    "dynamic_sampling/all_wrong_group_count": float(
                                        dynamic_filter_state["all_wrong_group_count"]
                                    ),
                                    "dynamic_sampling/accepted_group_ratio": float(accepted_group_count / generated_groups)
                                    if generated_groups
                                    else 0.0,
                                    "dynamic_sampling/all_correct_group_ratio": float(
                                        dynamic_filter_state["all_correct_group_count"] / generated_groups
                                    )
                                    if generated_groups
                                    else 0.0,
                                    "dynamic_sampling/all_wrong_group_ratio": float(
                                        dynamic_filter_state["all_wrong_group_count"] / generated_groups
                                    )
                                    if generated_groups
                                    else 0.0,
                                    "dynamic_sampling/rejected_group_ratio": float(rejected_groups / generated_groups)
                                    if generated_groups
                                    else 0.0,
                                    "dynamic_sampling/raw_rollout_count": float(generated_rollouts),
                                    "dynamic_sampling/accepted_rollout_count": float(accepted_rollouts),
                                    "dynamic_sampling/rejected_rollout_count": float(
                                        max(generated_rollouts - accepted_rollouts, 0)
                                    ),
                                    "dynamic_sampling/extra_rollout_count": float(max(generated_rollouts - accepted_rollouts, 0)),
                                    "dynamic_sampling/raw_response_tokens": float(generated_response_tokens),
                                    "dynamic_sampling/accepted_response_tokens": float(accepted_response_tokens),
                                    "dynamic_sampling/rejected_response_tokens": float(
                                        max(generated_response_tokens - accepted_response_tokens, 0)
                                    ),
                                }
                            )
                            candidate_group_count = generated_groups
                            candidate_batch_size = generated_rollouts
                            candidate_prompt_tokens = int(dynamic_filter_state["prompt_tokens_generated"])
                            candidate_response_tokens = generated_response_tokens
                            dynamic_filter_state = None
                        else:
                            metrics.update(compute_reward_extra_metrics(candidate_reward_extra_infos_dict))
                            metrics.update(
                                compute_group_metrics(
                                    uids=candidate_uids,
                                    raw_correctness=raw_correctness_values,
                                    expected_group_size=self.config.actor_rollout_ref.rollout.n,
                                )
                            )
                            metrics.update(
                                compute_length_metrics(
                                    response_lengths=candidate_response_lengths,
                                    raw_correctness=raw_correctness_values,
                                    max_response_length=batch.batch["responses"].shape[-1],
                                )
                            )
                            metrics.update(filter_group_metrics)
                            accepted_group_count = _count_prompt_groups(batch)
                            accepted_rollouts = len(batch)
                            accepted_response_tokens = int(
                                batch.batch["response_mask"].sum().detach().cpu().item()
                            )

                        if hive_post_result is None:
                            n_gpus_for_budget = self.resource_pool_manager.get_n_gpus()
                            metrics.update(
                                self._budget_tracker.update(
                                    candidate_prompt_groups_step=candidate_group_count,
                                    accepted_prompt_groups_step=accepted_group_count,
                                    responses_generated_step=candidate_batch_size,
                                    prompt_tokens_generated_step=candidate_prompt_tokens,
                                    response_tokens_generated_step=candidate_response_tokens,
                                    effective_prompt_groups_step=accepted_group_count,
                                    effective_responses_step=accepted_rollouts,
                                    effective_response_tokens_step=accepted_response_tokens,
                                    rollout_time_seconds_step=float(timing_raw.get("gen", 0.0)),
                                    optimizer_steps_step=1,
                                    n_gpus=n_gpus_for_budget,
                                )
                            )
                        target_response_tokens = self.config.trainer.get("target_response_tokens", None)
                        if target_response_tokens is not None and float(target_response_tokens) > 0.0:
                            target_response_tokens = float(target_response_tokens)
                            cumulative_response_tokens = float(
                                metrics.get("budget/response_tokens_generated_cumulative", 0.0)
                            )
                            remaining_response_tokens = max(target_response_tokens - cumulative_response_tokens, 0.0)
                            target_reached = cumulative_response_tokens >= target_response_tokens
                            metrics.update(
                                {
                                    "budget/target_response_tokens": target_response_tokens,
                                    "budget/target_response_tokens_remaining": remaining_response_tokens,
                                    "budget/target_response_tokens_reached": float(target_reached),
                                }
                            )
                            if target_reached:
                                is_last_step = True
                                self.total_training_steps = min(self.total_training_steps, self.global_steps)
                                try:
                                    progress_bar.total = self.global_steps
                                    progress_bar.refresh()
                                except Exception:
                                    pass
                                print(
                                    "Target response-token budget reached: "
                                    f"{cumulative_response_tokens:.0f} >= {target_response_tokens:.0f}; "
                                    f"stopping after global_step_{self.global_steps}."
                                )

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup > self.global_steps:
                        # Still in critic warmup, only update weights to wake up rollout replicas.
                        self.checkpoint_manager.update_weights(self.global_steps)
                        metrics.update(self._commit_hive_step(hive_stage1_selector, hive_pending_commit))
                    else:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Publish selector RNG, visits, and controller changes only after the optimizer operation.
                        metrics.update(self._commit_hive_step(hive_stage1_selector, hive_pending_commit))

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if _should_save_checkpoint(
                            trainer_config=self.config.trainer,
                            global_step=self.global_steps,
                            is_last_step=is_last_step,
                            esi=esi_close_to_expiration,
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    rollout_dump_interval = int(self.config.trainer.get("rollout_dump_interval", 1))
                    if (
                        rollout_data_dir
                        and rollout_dump_interval > 0
                        and self.global_steps % rollout_dump_interval == 0
                    ):
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                        if self.config.trainer.get("update_best_checkpoint_metadata", True):
                            val_metrics.update(self._update_best_checkpoint_metadata(val_metrics))
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # GDPO per-component reward metrics
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(
                    compute_section18_timing_metrics(
                        timing_raw=timing_raw,
                        stage1_seconds=float(metrics.get("hive/stage1_latency_seconds", 0.0)),
                        stage2_entropy_seconds=float(
                            metrics.get("hive/stage2_entropy_latency_seconds", 0.0)
                        ),
                        iteration_total_seconds=time.monotonic() - iteration_started_at,
                    )
                )
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                budget_snapshot = self._budget_tracker.snapshot(n_gpus=n_gpus)
                if self._hive_compute_counters is not None:
                    for compute_key in (
                        "compute/generated_prompt_groups",
                        "compute/generated_responses",
                        "compute/generated_response_tokens",
                        "compute/effective_prompt_groups",
                        "compute/effective_responses",
                        "compute/effective_response_tokens",
                    ):
                        if float(metrics[compute_key]) != float(budget_snapshot[compute_key]):
                            raise RuntimeError(
                                f"HIVE/shared compute accounting mismatch for {compute_key}: "
                                f"{metrics[compute_key]} != {budget_snapshot[compute_key]}"
                            )
                metrics.update(budget_snapshot)
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # Per-request spec decode metrics.
                metrics.update(
                    compute_spec_decode_metrics(
                        batch.non_tensor_batch.get("spec_num_draft_tokens", None),
                        batch.non_tensor_batch.get("spec_num_accepted_tokens", None),
                        batch.non_tensor_batch.get("spec_num_verify_steps", None),
                    )
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                SkipManager.set_step(self.global_steps)

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    self._shutdown_dump_executor()
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

        # Ensure dump executor is shut down when training loop ends without reaching is_last_step
        self._shutdown_dump_executor()
