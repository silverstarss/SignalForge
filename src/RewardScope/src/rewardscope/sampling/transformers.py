"""Optional Hugging Face Transformers sampler for decoder-only causal language models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from rewardscope.config import ModelConfig, SamplingConfig
from rewardscope.sampling.schema import GeneratedResponse


def build_generation_kwargs(config: SamplingConfig) -> dict[str, object]:
    """Build the version-stable subset of ``model.generate`` keyword arguments."""
    kwargs: dict[str, object] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.temperature > 0,
    }
    if config.temperature == 0:
        kwargs["num_beams"] = 1
        return kwargs

    kwargs.update(
        {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "num_return_sequences": config.num_samples,
        }
    )
    return kwargs


def clean_generated_token_ids(
    generated_ids: Sequence[int],
    *,
    eos_token_id: int | Sequence[int] | None,
    pad_token_id: int | None,
) -> tuple[list[int], Literal["eos", "length"]]:
    """Classify a raw continuation and remove its EOS and padding tokens."""
    ids = _require_token_ids("generated_ids", generated_ids)
    eos_ids = _normalize_optional_token_ids("eos_token_id", eos_token_id)
    if pad_token_id is not None:
        _require_token_id("pad_token_id", pad_token_id)

    eos_index = next((index for index, token_id in enumerate(ids) if token_id in eos_ids), None)
    continuation = ids[:eos_index] if eos_index is not None else ids
    clean_ids = [
        token_id
        for token_id in continuation
        if token_id not in eos_ids and token_id != pad_token_id
    ]
    return clean_ids, "eos" if eos_index is not None else "length"


def reshape_generated_responses(
    *,
    prompt_indices: Sequence[int],
    prompt_token_counts: Sequence[int],
    continuations: Sequence[tuple[str, Sequence[int], Literal["eos", "length"]]],
    num_samples: int,
) -> list[GeneratedResponse]:
    """Attach prompt/sample indexes to flat Transformer generation output."""
    _require_positive_int("num_samples", num_samples)
    if len(prompt_indices) != len(prompt_token_counts):
        raise ValueError("prompt_indices and prompt_token_counts must have equal lengths.")
    if len(continuations) != len(prompt_indices) * num_samples:
        raise ValueError("continuations must contain one item per prompt and sample.")

    responses: list[GeneratedResponse] = []
    for local_prompt_index, (prompt_index, prompt_tokens) in enumerate(
        zip(prompt_indices, prompt_token_counts, strict=True)
    ):
        _require_non_negative_int("prompt_index", prompt_index)
        _require_non_negative_int("prompt_tokens", prompt_tokens)
        for sample_index in range(num_samples):
            response, response_ids, finish_reason = continuations[
                local_prompt_index * num_samples + sample_index
            ]
            responses.append(
                GeneratedResponse(
                    prompt_index=prompt_index,
                    sample_index=sample_index,
                    response=response,
                    prompt_tokens=prompt_tokens,
                    response_tokens=len(_require_token_ids("response_ids", response_ids)),
                    finish_reason=finish_reason,
                )
            )
    return responses


class TransformersSampler:
    """Generate batched responses with a decoder-only Transformers model."""

    def __init__(self, model: Any, tokenizer: Any, model_config: ModelConfig) -> None:
        if not isinstance(model_config, ModelConfig):
            raise TypeError("model_config must be a ModelConfig.")
        if getattr(getattr(model, "config", None), "is_encoder_decoder", False):
            raise ValueError("Only decoder-only causal language models are supported.")

        self._model = model
        self._tokenizer = tokenizer
        self._model_config = model_config
        self._ensure_left_padding()
        self._ensure_pad_token()
        self._model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_config: ModelConfig,
        *,
        device_map: str | None = "auto",
        torch_dtype: str | object = "auto",
    ) -> TransformersSampler:
        """Load a causal LM and tokenizer without enabling remote code execution."""
        _require_model_dependencies()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer_name = model_config.tokenizer_name or model_config.name
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_config.name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=False,
        )
        return cls(model=model, tokenizer=tokenizer, model_config=model_config)

    def generate(
        self, prompts: Sequence[PromptInput], sampling_config: SamplingConfig
    ) -> list[GeneratedResponse]:
        """Generate ``num_samples`` responses per prompt in a stable flat order.

        Reproducibility is guaranteed only for identical library versions, device,
        prompt order, batch size, configuration, and generation seed. It is not guaranteed
        across different batch sizes.
        """
        torch = _require_model_dependencies()
        if not isinstance(sampling_config, SamplingConfig):
            raise TypeError("sampling_config must be a SamplingConfig.")
        prompt_list = _validate_prompts(prompts)
        if not prompt_list:
            return []

        context_window = self._resolve_context_window()
        generation_kwargs = build_generation_kwargs(sampling_config)
        responses: list[GeneratedResponse] = []
        rng_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

        with torch.random.fork_rng(devices=rng_devices), torch.inference_mode():
            torch.manual_seed(sampling_config.generation_seed)
            for start in range(0, len(prompt_list), sampling_config.batch_size):
                batch_prompts = prompt_list[start : start + sampling_config.batch_size]
                tokenized = self._tokenize(batch_prompts)
                input_ids = tokenized["input_ids"]
                attention_mask = tokenized["attention_mask"]
                prompt_token_counts = [int(value) for value in attention_mask.sum(dim=1).tolist()]
                self._validate_context_window(
                    prompt_token_counts,
                    sampling_config.max_new_tokens,
                    context_window,
                )

                input_ids = self._move_to_model_device(input_ids)
                attention_mask = self._move_to_model_device(attention_mask)
                generated = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=self._tokenizer.pad_token_id,
                    **generation_kwargs,
                )
                generated_rows = generated.tolist()
                padded_input_width = int(input_ids.shape[1])
                continuations: list[tuple[str, Sequence[int], Literal["eos", "length"]]] = []
                for generated_row in generated_rows:
                    raw_ids = generated_row[padded_input_width:]
                    clean_ids, finish_reason = clean_generated_token_ids(
                        raw_ids,
                        eos_token_id=self._eos_token_id(),
                        pad_token_id=self._tokenizer.pad_token_id,
                    )
                    continuations.append(
                        (
                            self._tokenizer.decode(clean_ids, skip_special_tokens=True),
                            clean_ids,
                            finish_reason,
                        )
                    )

                responses.extend(
                    reshape_generated_responses(
                        prompt_indices=range(start, start + len(batch_prompts)),
                        prompt_token_counts=prompt_token_counts,
                        continuations=continuations,
                        num_samples=sampling_config.num_samples,
                    )
                )
        return responses

    def render_prompt(self, prompt: PromptInput) -> str:
        """Render one dataset prompt into the exact model input text without sampling."""
        prompt_list = _validate_prompts([prompt])
        if self._resolve_prompt_format() == "plain":
            if not isinstance(prompt_list[0], str):
                raise ValueError("Plain prompt format does not support role-separated messages.")
            return prompt_list[0]
        rendered = self._tokenizer.apply_chat_template(
            _messages_for_prompt(prompt_list[0]),
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str) or not rendered:
            raise ValueError("Tokenizer chat template must render one prompt as a non-empty string.")
        return rendered

    def _tokenize(self, prompts: list[PromptInput]) -> Mapping[str, Any]:
        prompt_format = self._resolve_prompt_format()
        if prompt_format == "chat":
            messages = [_messages_for_prompt(prompt) for prompt in prompts]
            tokenized = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                padding=True,
                return_tensors="pt",
            )
        else:
            if any(not isinstance(prompt, str) for prompt in prompts):
                raise ValueError("Plain prompt format does not support role-separated messages.")
            tokenized = self._tokenizer(
                prompts,
                padding=True,
                return_tensors="pt",
                truncation=False,
            )

        if not isinstance(tokenized, Mapping):
            raise ValueError("Tokenizer must return a mapping with input_ids and attention_mask.")
        if "input_ids" not in tokenized or "attention_mask" not in tokenized:
            raise ValueError("Tokenizer output must include input_ids and attention_mask.")
        return tokenized

    def _resolve_prompt_format(self) -> Literal["chat", "plain"]:
        configured = self._model_config.prompt_format
        if configured == "auto":
            chat_template = getattr(self._tokenizer, "chat_template", None)
            return "chat" if isinstance(chat_template, str) and chat_template.strip() else "plain"
        return configured

    def _resolve_context_window(self) -> int:
        if self._model_config.context_window is not None:
            return self._model_config.context_window
        context_window = getattr(getattr(self._model, "config", None), "max_position_embeddings", None)
        if isinstance(context_window, int) and not isinstance(context_window, bool) and context_window > 0:
            return context_window
        raise ValueError(
            "Model context window is unknown; set model.context_window explicitly."
        )

    def _validate_context_window(
        self,
        prompt_token_counts: Sequence[int],
        max_new_tokens: int,
        context_window: int,
    ) -> None:
        for prompt_tokens in prompt_token_counts:
            if prompt_tokens + max_new_tokens > context_window:
                raise ValueError(
                    "prompt_tokens + max_new_tokens exceeds the model context window "
                    f"({prompt_tokens} + {max_new_tokens} > {context_window})."
                )

    def _ensure_left_padding(self) -> None:
        self._tokenizer.padding_side = "left"

    def _ensure_pad_token(self) -> None:
        if self._tokenizer.pad_token_id is not None:
            return
        eos_token = getattr(self._tokenizer, "eos_token", None)
        if not isinstance(eos_token, str) or not eos_token:
            raise ValueError("Tokenizer needs a pad token or an eos_token to use as padding.")
        self._tokenizer.pad_token = eos_token
        if self._tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer could not assign a pad token from its eos_token.")

    def _eos_token_id(self) -> int | Sequence[int] | None:
        generation_config = getattr(self._model, "generation_config", None)
        eos_token_id = getattr(generation_config, "eos_token_id", None)
        if eos_token_id is not None:
            return eos_token_id
        return getattr(self._tokenizer, "eos_token_id", None)

    def _move_to_model_device(self, tensor: Any) -> Any:
        device = getattr(self._model, "device", None)
        return tensor.to(device) if device is not None else tensor


def _require_model_dependencies() -> Any:
    try:
        import torch
        import transformers  # noqa: F401
    except ModuleNotFoundError as error:
        raise RuntimeError(
            'Model sampling requires optional dependencies. Run: pip install -e ".[model]"'
        ) from error
    return torch


PromptInput = str | Sequence[Mapping[str, str]]


def _validate_prompts(prompts: Sequence[PromptInput]) -> list[PromptInput]:
    if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a sequence of strings or role-separated messages.")
    prompt_list = list(prompts)
    for prompt in prompt_list:
        if isinstance(prompt, str):
            if not prompt:
                raise ValueError("prompt strings must be non-empty.")
            continue
        _messages_for_prompt(prompt)
    return prompt_list


def _messages_for_prompt(prompt: PromptInput) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, (str, bytes)) or not isinstance(prompt, Sequence) or not prompt:
        raise ValueError("Role-separated messages must be a non-empty sequence.")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(prompt):
        if not isinstance(message, Mapping):
            raise ValueError(f"messages[{index}] must be a mapping.")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}] must contain a supported role and non-empty content.")
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] != "user":
        raise ValueError("Role-separated messages must end with a user message for generation.")
    return messages


def _normalize_optional_token_ids(
    name: str, value: int | Sequence[int] | None
) -> frozenset[int]:
    if value is None:
        return frozenset()
    if isinstance(value, int) and not isinstance(value, bool):
        return frozenset({value})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return frozenset(_require_token_ids(name, value))
    raise ValueError(f"{name} must be an integer, a sequence of integers, or None.")


def _require_token_ids(name: str, value: Sequence[int]) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of integers.")
    token_ids = list(value)
    for token_id in token_ids:
        _require_token_id(name, token_id)
    return token_ids


def _require_token_id(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must contain only integers.")


def _require_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _require_non_negative_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
