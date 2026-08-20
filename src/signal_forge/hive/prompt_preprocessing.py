"""Canonical text-prompt preprocessing shared with the veRL AgentLoop."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizerBase

from signal_forge.hive.identity import PromptIdentityError
from verl.utils import tensordict_utils as tu
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tokenizer.chat_template import preprocess_chat_prompt


@dataclass(frozen=True)
class CanonicalHivePrompt:
    """Stable identity plus the unpadded tokens that condition rollout generation."""

    prompt_id: str
    input_ids: tuple[int, ...]
    untruncated_token_count: int
    left_truncated_token_count: int

    @property
    def valid_token_count(self) -> int:
        return len(self.input_ids)


class HivePromptPreprocessor:
    """Bind stable IDs to the canonical veRL text-prompt tokenization path."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_prompt_length: int,
        apply_chat_template_kwargs: Mapping[str, Any] | None = None,
    ):
        if not hasattr(tokenizer, "apply_chat_template"):
            raise TypeError("tokenizer must expose apply_chat_template")
        if isinstance(max_prompt_length, bool) or not isinstance(max_prompt_length, int) or max_prompt_length <= 0:
            raise ValueError("max_prompt_length must be a positive integer")
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.apply_chat_template_kwargs = dict(apply_chat_template_kwargs or {})

    def preprocess(self, prompt_id: str, raw_prompt: Sequence[Mapping[str, Any]]) -> CanonicalHivePrompt:
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise PromptIdentityError("prompt_id must be a non-empty stable string")
        if isinstance(raw_prompt, (str, bytes)) or not isinstance(raw_prompt, Sequence) or not raw_prompt:
            raise ValueError("raw_prompt must be a non-empty sequence of chat messages")
        messages = []
        for index, message in enumerate(raw_prompt):
            if not isinstance(message, Mapping):
                raise ValueError(f"raw_prompt message at index {index} must be a mapping")
            messages.append(dict(message))

        prompt = preprocess_chat_prompt(
            self.tokenizer,
            messages,
            max_prompt_length=self.max_prompt_length,
            apply_chat_template_kwargs=self.apply_chat_template_kwargs,
        )
        return CanonicalHivePrompt(
            prompt_id=prompt_id,
            input_ids=prompt.input_ids,
            untruncated_token_count=prompt.untruncated_token_count,
            left_truncated_token_count=prompt.left_truncated_token_count,
        )

    def preprocess_batch(
        self,
        prompt_ids: Sequence[str],
        raw_prompts: Sequence[Sequence[Mapping[str, Any]]],
    ) -> tuple[CanonicalHivePrompt, ...]:
        if len(prompt_ids) != len(raw_prompts):
            raise ValueError("prompt_ids and raw_prompts must have the same length")
        if len(set(prompt_ids)) != len(prompt_ids):
            raise PromptIdentityError("prompt preprocessing batch contains duplicate prompt_id values")
        return tuple(
            self.preprocess(prompt_id, raw_prompt)
            for prompt_id, raw_prompt in zip(prompt_ids, raw_prompts, strict=True)
        )

    def build_entropy_rpc_batch(
        self,
        prompts: Sequence[CanonicalHivePrompt],
        *,
        pad_to_length: int | None = None,
    ) -> TensorDict | None:
        """Build prompt-only actor inputs, or return None without issuing an empty Ray RPC."""
        prompts = tuple(prompts)
        if not prompts:
            return None
        if any(not isinstance(prompt, CanonicalHivePrompt) for prompt in prompts):
            raise TypeError("prompts must contain CanonicalHivePrompt records")

        prompt_ids = [prompt.prompt_id for prompt in prompts]
        if len(set(prompt_ids)) != len(prompt_ids):
            raise PromptIdentityError("entropy batch contains duplicate prompt_id values")
        maximum_valid_length = max(prompt.valid_token_count for prompt in prompts)
        target_length = maximum_valid_length if pad_to_length is None else pad_to_length
        if isinstance(target_length, bool) or not isinstance(target_length, int) or target_length <= 0:
            raise ValueError("pad_to_length must be a positive integer or None")
        if target_length < maximum_valid_length:
            raise ValueError("pad_to_length cannot truncate canonical prompt tokens")

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id is required for actor entropy batching")

        input_rows = []
        attention_rows = []
        for prompt in prompts:
            padding = target_length - prompt.valid_token_count
            input_rows.append([pad_token_id] * padding + list(prompt.input_ids))
            attention_rows.append([False] * padding + [True] * prompt.valid_token_count)

        input_ids = torch.tensor(input_rows, dtype=torch.long)
        attention_mask = torch.tensor(attention_rows, dtype=torch.bool)
        position_ids = compute_position_id_with_mask(attention_mask.long())
        return tu.get_tensordict(
            tensor_dict={
                "prompt_id": prompt_ids,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompt_token_mask": attention_mask.clone(),
                "prompt_length": torch.tensor(
                    [prompt.valid_token_count for prompt in prompts], dtype=torch.int64
                ),
                "untruncated_prompt_length": torch.tensor(
                    [prompt.untruncated_token_count for prompt in prompts], dtype=torch.int64
                ),
                "left_truncated_token_count": torch.tensor(
                    [prompt.left_truncated_token_count for prompt in prompts], dtype=torch.int64
                ),
            }
        )


def validate_hive_prompt_preprocessing_scope(config: Any) -> None:
    """Reject HIVE execution modes outside the approved single-GPU text path."""
    trainer = _cfg_get(config, "trainer", None)
    nnodes = int(_cfg_get(trainer, "nnodes", 1))
    n_gpus_per_node = int(_cfg_get(trainer, "n_gpus_per_node", 1))
    if nnodes != 1 or n_gpus_per_node != 1:
        raise ValueError("HIVE prompt preprocessing currently supports single GPU execution only")

    actor_rollout_ref = _cfg_get(config, "actor_rollout_ref", None)
    actor = _cfg_get(actor_rollout_ref, "actor", None)
    if bool(_cfg_get(actor, "use_fused_kernels", False)):
        raise ValueError("HIVE prompt entropy does not support fused actor kernels")
    if int(_cfg_get(actor, "ulysses_sequence_parallel_size", 1)) != 1:
        raise ValueError("HIVE prompt entropy requires Ulysses sequence parallel size 1")

    data = _cfg_get(config, "data", None)
    continuous_token = _cfg_get(data, "continuous_token", None)
    if bool(_cfg_get(continuous_token, "enable", False)):
        raise ValueError("HIVE canonical preprocessing does not support continuous-token mode")


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)
