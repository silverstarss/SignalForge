"""Smoke-check this veRL reward manager against the Math-Verify adapter."""

from __future__ import annotations

import json

import numpy as np
import torch

from signal_forge.rewards.math_verify_adapter import compute_score
from verl import DataProto
from verl.workers.reward_manager.naive import NaiveRewardManager


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        ids = tuple(int(x) for x in token_ids.tolist())
        if ids == (11, 12, 13):
            return "PROMPT_SENTINEL: do not pass this text to reward."
        if ids == (21, 22, 23):
            return "Reasoning only in the response. The final answer is \\boxed{2}."
        raise AssertionError(f"Unexpected ids decoded by fake tokenizer: {ids}")


def _checked_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    assert "PROMPT_SENTINEL" not in solution_str, "solution_str unexpectedly contains prompt text"
    return compute_score(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info)


def main() -> None:
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[11, 12, 13]], dtype=torch.long),
            "responses": torch.tensor([[21, 22, 23]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
        },
        non_tensors={
            "data_source": np.array(["gsm8k"], dtype=object),
            "reward_model": np.array([{"style": "rule", "ground_truth": "2"}], dtype=object),
            "extra_info": np.array([{"prompt_id": "reward-manager-check", "source_dataset": "gsm8k"}], dtype=object),
        },
    )

    manager = NaiveRewardManager(tokenizer=_FakeTokenizer(), num_examine=0, compute_score=_checked_compute_score)
    output = manager(data, return_dict=True)
    reward_tensor = output["reward_tensor"]
    extra = output["reward_extra_info"]

    assert reward_tensor.shape == (1, 3), reward_tensor.shape
    assert float(reward_tensor[0, 0]) == 0.0
    assert float(reward_tensor[0, 1]) == 0.0
    assert float(reward_tensor[0, 2]) == 1.0, reward_tensor
    for key in [
        "score",
        "reward",
        "raw_correctness",
        "extracted",
        "correct",
        "extraction_ok",
        "format_ok",
        "verification_status",
    ]:
        assert key in extra, f"reward extra info missing {key}"
        assert len(extra[key]) == 1, f"reward extra info {key} length mismatch"

    print(
        json.dumps(
            {
                "reward_tensor": reward_tensor.tolist(),
                "reward_extra_keys": sorted(extra.keys()),
                "score": extra["score"][0],
                "verification_status": extra["verification_status"][0],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
