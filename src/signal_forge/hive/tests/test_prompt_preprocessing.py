from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from signal_forge.hive.identity import PromptIdentityError
from signal_forge.hive.prompt_preprocessing import (
    HivePromptPreprocessor,
    validate_hive_prompt_preprocessing_scope,
)
from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import normalize_token_ids
from verl.utils.tokenizer.chat_template import apply_chat_template
from verl.workers.rollout.replica import TokenOutput


QWEN25_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
REAL_QWEN25_TOKENIZER_DIR = os.environ.get("QWEN25_3B_LOCAL_DIR")
QWEN25_TEXT_CHAT_TEMPLATE = """{%- if messages[0]['role'] == 'system' -%}
{{ '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
{%- else -%}
{{ '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}
{%- endif -%}
{%- for message in messages -%}
{%- if not (loop.first and message['role'] == 'system') -%}
{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{ '<|im_start|>assistant\\n' }}
{%- endif -%}"""


@pytest.fixture
def qwen25_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "<|endoftext|>": 1,
                "<|im_start|>": 2,
                "<|im_end|>": 3,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        bos_token="<|endoftext|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
        chat_template=QWEN25_TEXT_CHAT_TEMPLATE,
    )
    tokenizer.padding_side = "left"
    return tokenizer


class CapturingServer:
    def __init__(self):
        self.prompt_ids: list[int] | None = None

    async def generate(self, *, prompt_ids: list[int], **kwargs: Any) -> TokenOutput:
        del kwargs
        self.prompt_ids = list(prompt_ids)
        return TokenOutput(token_ids=[91, 92], log_probs=[-0.1, -0.2])


def _agent_loop(tokenizer, *, prompt_length: int) -> tuple[SingleTurnAgentLoop, CapturingServer]:
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": prompt_length,
                    "response_length": 8,
                    "full_determinism": True,
                },
                "model": {},
            },
            "data": {
                "apply_chat_template_kwargs": {},
                "mm_processor_kwargs": {},
                "continuous_token": {"enable": False, "model_family": "auto"},
            },
        }
    )
    server = CapturingServer()
    loop = SingleTurnAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=server,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
    )
    return loop, server


async def _run_rollout(tokenizer, raw_prompt, *, prompt_length: int):
    agent_loop, server = _agent_loop(tokenizer, prompt_length=prompt_length)
    output = await agent_loop.run(sampling_params={}, raw_prompt=raw_prompt)
    return output, server


async def _render_incremental_without_system(tokenizer, messages, *, prompt_length: int):
    agent_loop, _ = _agent_loop(tokenizer, prompt_length=prompt_length)
    token_ids = await agent_loop.apply_chat_template(messages, remove_system_prompt=True)
    return token_ids, list(agent_loop.system_prompt)


def test_stage2_tokens_equal_actual_rollout_conditioning_tokens(qwen25_tokenizer):
    raw_prompt = [
        {"role": "system", "content": "Solve carefully and preserve exact notation."},
        {"role": "user", "content": "Solve 1 + 1 and return a boxed answer."},
    ]
    preprocessor = HivePromptPreprocessor(qwen25_tokenizer, max_prompt_length=128)
    stage2_prompt = preprocessor.preprocess("gsm8k:train:000001", raw_prompt)
    rollout_output, server = asyncio.run(
        _run_rollout(qwen25_tokenizer, raw_prompt, prompt_length=128)
    )

    assert server.prompt_ids == list(stage2_prompt.input_ids)
    assert rollout_output.prompt_ids == list(stage2_prompt.input_ids)
    assert stage2_prompt.prompt_id == "gsm8k:train:000001"


@pytest.mark.skipif(
    not REAL_QWEN25_TOKENIZER_DIR or not Path(REAL_QWEN25_TOKENIZER_DIR).is_dir(),
    reason="QWEN25_3B_LOCAL_DIR does not contain a local Qwen2.5-3B tokenizer",
)
def test_real_qwen25_tokenizer_matches_rollout_conditioning_tokens():
    tokenizer = AutoTokenizer.from_pretrained(
        REAL_QWEN25_TOKENIZER_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    raw_prompt = [{"role": "user", "content": "Solve 2 + 3 and return a boxed answer."}]
    stage2_prompt = HivePromptPreprocessor(tokenizer, max_prompt_length=128).preprocess(
        "gsm8k:train:real-tokenizer", raw_prompt
    )

    _, server = asyncio.run(_run_rollout(tokenizer, raw_prompt, prompt_length=128))

    assert server.prompt_ids == list(stage2_prompt.input_ids)


def test_qwen25_system_and_generation_prompt_suffix_are_exact(qwen25_tokenizer):
    messages = [{"role": "user", "content": "Question"}]
    rendered = apply_chat_template(
        qwen25_tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    assert rendered.startswith(f"<|im_start|>system\n{QWEN25_DEFAULT_SYSTEM}<|im_end|>\n")
    assert rendered.endswith("<|im_start|>assistant\n")

    explicit_system = "Answer only in LaTeX."
    explicit_rendered = apply_chat_template(
        qwen25_tokenizer,
        [{"role": "system", "content": explicit_system}, *messages],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert explicit_rendered.startswith(f"<|im_start|>system\n{explicit_system}<|im_end|>\n")
    assert QWEN25_DEFAULT_SYSTEM not in explicit_rendered


def test_qwen25_bos_eos_and_generation_suffix_token_ids(qwen25_tokenizer):
    prompt = HivePromptPreprocessor(qwen25_tokenizer, max_prompt_length=128).preprocess(
        "prompt:1", [{"role": "user", "content": "Question"}]
    )
    suffix_ids = qwen25_tokenizer(
        "<|im_start|>assistant\n",
        add_special_tokens=False,
    )["input_ids"]

    assert list(prompt.input_ids[-len(suffix_ids) :]) == suffix_ids
    assert prompt.input_ids[0] == qwen25_tokenizer.convert_tokens_to_ids("<|im_start|>")
    assert prompt.input_ids[0] != qwen25_tokenizer.bos_token_id
    assert prompt.input_ids[-1] != qwen25_tokenizer.eos_token_id
    assert qwen25_tokenizer.eos_token_id in prompt.input_ids


def test_truncation_matches_rollout_and_accounts_for_original_length(qwen25_tokenizer):
    raw_prompt = [{"role": "user", "content": " ".join(["term"] * 80)}]
    direct_ids = normalize_token_ids(
        apply_chat_template(
            qwen25_tokenizer,
            raw_prompt,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    max_prompt_length = 20
    prompt = HivePromptPreprocessor(qwen25_tokenizer, max_prompt_length=max_prompt_length).preprocess(
        "prompt:long", raw_prompt
    )
    _, server = asyncio.run(
        _run_rollout(qwen25_tokenizer, raw_prompt, prompt_length=max_prompt_length)
    )

    assert list(prompt.input_ids) == direct_ids[-max_prompt_length:]
    assert server.prompt_ids == direct_ids[-max_prompt_length:]
    assert prompt.valid_token_count == max_prompt_length
    assert prompt.untruncated_token_count == len(direct_ids)
    assert prompt.left_truncated_token_count == len(direct_ids) - max_prompt_length


def test_entropy_batch_has_no_responses_and_padding_does_not_change_canonical_tokens(qwen25_tokenizer):
    preprocessor = HivePromptPreprocessor(qwen25_tokenizer, max_prompt_length=128)
    short = preprocessor.preprocess("prompt:short", [{"role": "user", "content": "Short"}])
    long = preprocessor.preprocess(
        "prompt:long",
        [{"role": "user", "content": "A substantially longer prompt for padding."}],
    )
    short_alone = preprocessor.build_entropy_rpc_batch([short])
    mixed = preprocessor.build_entropy_rpc_batch([short, long], pad_to_length=64)

    assert short_alone is not None
    assert mixed is not None
    alone_ids = short_alone["input_ids"][0][short_alone["attention_mask"][0]].tolist()
    mixed_ids = mixed["input_ids"][0][mixed["attention_mask"][0]].tolist()
    assert alone_ids == mixed_ids == list(short.input_ids)
    assert tu.get(mixed, "prompt_id") == ["prompt:short", "prompt:long"]
    assert mixed["prompt_token_mask"].equal(mixed["attention_mask"])
    assert mixed["prompt_length"].tolist() == [short.valid_token_count, long.valid_token_count]
    for index, prompt in enumerate((short, long)):
        valid_positions = mixed["position_ids"][index][mixed["attention_mask"][index]].tolist()
        assert valid_positions == list(range(prompt.valid_token_count))
    assert "responses" not in mixed
    assert "response_mask" not in mixed


def test_empty_candidates_short_circuit_before_actor_rpc(qwen25_tokenizer):
    preprocessor = HivePromptPreprocessor(qwen25_tokenizer, max_prompt_length=128)

    assert preprocessor.build_entropy_rpc_batch([]) is None


@pytest.mark.parametrize("prompt_id", [None, "", "   ", 7])
def test_missing_or_malformed_prompt_id_is_rejected(qwen25_tokenizer, prompt_id):
    preprocessor = HivePromptPreprocessor(qwen25_tokenizer, max_prompt_length=128)

    with pytest.raises(PromptIdentityError, match="prompt_id"):
        preprocessor.preprocess(prompt_id, [{"role": "user", "content": "Question"}])


def test_remove_system_prompt_keeps_existing_remove_then_truncate_order(qwen25_tokenizer):
    messages = [{"role": "user", "content": " ".join(["tool"] * 30)}]
    full_ids = normalize_token_ids(
        apply_chat_template(
            qwen25_tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    prompt_length = 12

    actual, system_prompt = asyncio.run(
        _render_incremental_without_system(
            qwen25_tokenizer, messages, prompt_length=prompt_length
        )
    )

    assert actual == full_ids[len(system_prompt) :][-prompt_length:]


def test_hive_disabled_rollout_tokenization_is_unchanged(qwen25_tokenizer):
    raw_prompt = [{"role": "user", "content": "Baseline GRPO prompt"}]
    expected = normalize_token_ids(
        apply_chat_template(
            qwen25_tokenizer,
            raw_prompt,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    _, server = asyncio.run(
        _run_rollout(qwen25_tokenizer, raw_prompt, prompt_length=128)
    )

    assert server.prompt_ids == expected


def test_hive_scope_validation_accepts_only_current_single_gpu_path():
    supported = OmegaConf.create(
        {
            "trainer": {"nnodes": 1, "n_gpus_per_node": 1},
            "actor_rollout_ref": {
                "actor": {"use_fused_kernels": False, "ulysses_sequence_parallel_size": 1},
            },
            "data": {"continuous_token": {"enable": False}},
        }
    )
    validate_hive_prompt_preprocessing_scope(supported)

    for path, value, message in (
        ("trainer.n_gpus_per_node", 2, "single GPU"),
        ("actor_rollout_ref.actor.use_fused_kernels", True, "fused"),
        ("actor_rollout_ref.actor.ulysses_sequence_parallel_size", 2, "Ulysses"),
        ("data.continuous_token.enable", True, "continuous-token"),
    ):
        unsupported = OmegaConf.create(OmegaConf.to_container(supported, resolve=True))
        OmegaConf.update(unsupported, path, value)
        with pytest.raises(ValueError, match=message):
            validate_hive_prompt_preprocessing_scope(unsupported)
