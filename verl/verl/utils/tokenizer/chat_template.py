# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase, ProcessorMixin

from .tokenizer import normalize_token_ids

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass(frozen=True)
class CanonicalChatPrompt:
    """Unpadded prompt tokens produced by the rollout chat-template path."""

    input_ids: tuple[int, ...]
    untruncated_token_count: int
    left_truncated_token_count: int

    @property
    def valid_token_count(self) -> int:
        return len(self.input_ids)


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    token1 = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True, **apply_chat_template_kwargs
        )
    )
    token2 = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}] * 2,
            add_generation_prompt=False,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    return system_prompt


def extract_system_prompt_and_generation(tokenizer, **apply_chat_template_kwargs):
    token1 = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True, **apply_chat_template_kwargs
        )
    )
    token2 = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}] * 2,
            add_generation_prompt=False,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True, **apply_chat_template_kwargs
        )
    )
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt


def apply_chat_template(
    processor: PreTrainedTokenizerBase | ProcessorMixin,
    messages: list[dict],
    *,
    tokenize: bool = True,
    add_generation_prompt: bool = True,
    tools=None,
    return_dict: bool = False,
    **kwargs,
) -> list[int] | str:
    """apply_chat_template to messages with special attention to template requiring
    at least one user message, e.g. Qwen3.5.

    Args:
        processor: tokenizer or processor.
        messages: list[dict], messages.
        tokenize: bool, whether to tokenize the output.
        add_generation_prompt: bool, whether to add generation prompt.
        tools: list[dict], tools schema.
        return_dict: bool, whether to return a dict.
        **kwargs: additional arguments for apply_chat_template.

    Returns:
        list[int] | str: tokenized ids or text string.
    """
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
    except Exception:
        # Qwen3.5 apply_chat_template needs messages with at least one user message
        dummy_user_message = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        dummy_user_prefix = processor.apply_chat_template(
            dummy_user_message,
            tokenize=tokenize,
            add_generation_prompt=False,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
        output = processor.apply_chat_template(
            dummy_user_message + messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )

        if not tokenize:  # tokenize=False
            return output[len(dummy_user_prefix) :]
        elif not return_dict:  # tokenize=True and return_dict=False
            if isinstance(output[0], list):  # transformers>=5
                assert len(output) == 1, "output must be a list[int] or list[list[int]]"
                dummy_user_prefix = dummy_user_prefix[0]
                output = output[0]
            return output[len(dummy_user_prefix) :]
        else:  # tokenize=True and return_dict=True and return_tensors="pt"
            dummy_user_prefix = dict(dummy_user_prefix)
            output = dict(output)
            prefix_len = dummy_user_prefix["input_ids"].shape[1]
            output["input_ids"] = output["input_ids"][:, prefix_len:]
            output["attention_mask"] = output["attention_mask"][:, prefix_len:]
            if "mm_token_type_ids" in output:
                output["mm_token_type_ids"] = output["mm_token_type_ids"][:, prefix_len:]
            return output


def preprocess_chat_prompt(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    *,
    max_prompt_length: int | None,
    tools=None,
    apply_chat_template_kwargs: dict | None = None,
) -> CanonicalChatPrompt:
    """Apply the rollout template and its text-only left-truncation semantics."""
    if max_prompt_length is not None and (
        isinstance(max_prompt_length, bool)
        or not isinstance(max_prompt_length, int)
        or max_prompt_length <= 0
    ):
        raise ValueError("max_prompt_length must be a positive integer or None")

    tokenized_prompt = apply_chat_template(
        tokenizer,
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=True,
        **dict(apply_chat_template_kwargs or {}),
    )
    prompt_ids = normalize_token_ids(tokenized_prompt)
    untruncated_token_count = len(prompt_ids)
    left_truncated_token_count = (
        0 if max_prompt_length is None else max(0, untruncated_token_count - max_prompt_length)
    )
    if left_truncated_token_count:
        logger.warning(
            "Prompt of %d tokens exceeds rollout.prompt_length=%d; left-truncating.",
            untruncated_token_count,
            max_prompt_length,
        )
        prompt_ids = prompt_ids[-max_prompt_length:]

    return CanonicalChatPrompt(
        input_ids=tuple(prompt_ids),
        untruncated_token_count=untruncated_token_count,
        left_truncated_token_count=left_truncated_token_count,
    )
