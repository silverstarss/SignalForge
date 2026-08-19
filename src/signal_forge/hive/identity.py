"""Stable prompt identity plumbing for HIVE."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any

import numpy as np


class PromptIdentityError(ValueError):
    """Raised when a batch cannot be mapped to stable dataset prompt IDs."""


def extract_stable_prompt_ids(extra_infos: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    """Extract unique, explicit prompt IDs without generating positional fallbacks."""
    prompt_ids: list[str] = []
    for index, extra_info in enumerate(extra_infos):
        if not isinstance(extra_info, Mapping):
            raise PromptIdentityError(f"extra_info at batch index {index} must be a mapping")
        if "prompt_id" not in extra_info:
            raise PromptIdentityError(f"extra_info at batch index {index} is missing prompt_id")

        prompt_id = extra_info["prompt_id"]
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise PromptIdentityError(f"prompt_id at batch index {index} must be a non-empty string")
        prompt_ids.append(prompt_id)

    if len(set(prompt_ids)) != len(prompt_ids):
        duplicates = sorted({prompt_id for prompt_id in prompt_ids if prompt_ids.count(prompt_id) > 1})
        raise PromptIdentityError(f"duplicate prompt_id values in raw prompt batch: {duplicates[:5]}")
    return tuple(prompt_ids)


def attach_stable_prompt_ids(non_tensor_batch: MutableMapping[str, Any]) -> tuple[str, ...]:
    """Promote ``extra_info.prompt_id`` to a top-level non-tensor batch field."""
    if "extra_info" not in non_tensor_batch:
        raise PromptIdentityError("non-tensor batch is missing extra_info required for stable prompt identity")

    prompt_ids = extract_stable_prompt_ids(non_tensor_batch["extra_info"])
    if "prompt_id" in non_tensor_batch:
        existing = tuple(np.asarray(non_tensor_batch["prompt_id"], dtype=object).tolist())
        if existing != prompt_ids:
            raise PromptIdentityError("existing prompt_id field does not match extra_info.prompt_id")

    non_tensor_batch["prompt_id"] = np.asarray(prompt_ids, dtype=object)
    return prompt_ids
