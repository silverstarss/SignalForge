from __future__ import annotations

import numpy as np
import pytest

from signal_forge.hive.identity import PromptIdentityError, attach_stable_prompt_ids, extract_stable_prompt_ids


def test_extract_stable_prompt_ids_preserves_explicit_ids_across_reordering():
    extra_infos = [
        {"prompt_id": "gsm8k:openai-main:train:000007"},
        {"prompt_id": "math:competition_math:train:algebra:12"},
    ]

    assert extract_stable_prompt_ids(extra_infos) == (
        "gsm8k:openai-main:train:000007",
        "math:competition_math:train:algebra:12",
    )
    assert extract_stable_prompt_ids(reversed(extra_infos)) == (
        "math:competition_math:train:algebra:12",
        "gsm8k:openai-main:train:000007",
    )


@pytest.mark.parametrize(
    "extra_infos, message",
    [
        ([{}], "missing prompt_id"),
        ([{"prompt_id": "  "}], "non-empty string"),
        ([{"prompt_id": 12}], "non-empty string"),
        ([{"prompt_id": "same"}, {"prompt_id": "same"}], "duplicate prompt_id"),
    ],
)
def test_extract_stable_prompt_ids_rejects_unstable_or_ambiguous_identity(extra_infos, message):
    with pytest.raises(PromptIdentityError, match=message):
        extract_stable_prompt_ids(extra_infos)


def test_attach_stable_prompt_ids_promotes_extra_info_without_rewriting_it():
    non_tensor_batch = {
        "extra_info": np.asarray([{"prompt_id": "p:1"}, {"prompt_id": "p:2"}], dtype=object),
    }

    attach_stable_prompt_ids(non_tensor_batch)

    assert non_tensor_batch["prompt_id"].tolist() == ["p:1", "p:2"]
    assert non_tensor_batch["extra_info"][0]["prompt_id"] == "p:1"


def test_attach_stable_prompt_ids_rejects_conflicting_existing_field():
    non_tensor_batch = {
        "extra_info": np.asarray([{"prompt_id": "p:1"}], dtype=object),
        "prompt_id": np.asarray(["different"], dtype=object),
    }

    with pytest.raises(PromptIdentityError, match="does not match"):
        attach_stable_prompt_ids(non_tensor_batch)
