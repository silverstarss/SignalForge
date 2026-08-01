"""Dataset adapters that produce RewardScope examples."""

from rewardscope.datasets.gsm8k import (
    DEFAULT_GSM8K_PROMPT_TEMPLATE,
    GSM8K_COT_4SHOT_PROMPT_TEMPLATE,
    GSM8K_COT_4SHOT_TERMINAL_PROMPT_TEMPLATE,
    GSM8K_COT_4SHOT_MULTITURN_TERMINAL_TARGET_TEMPLATE,
    GSM8K_ZERO_SHOT_BOXED_PROMPT_TEMPLATE,
    build_gsm8k_cot_4shot_multiturn_terminal_messages,
    STRICT_GSM8K_PROMPT_TEMPLATE,
    load_gsm8k_examples,
)
from rewardscope.datasets.load import load_dataset_examples, load_dataset_result
from rewardscope.datasets.math import (
    MATH_CONFIGS,
    MATH_DATASET_ID,
    MODELSCOPE_MATH_DATASET_ID,
    MATH_ZERO_SHOT_BOXED_PROMPT_TEMPLATE,
    load_math_examples,
)
from rewardscope.datasets.schema import ChatMessage, DatasetExample, DatasetLoadResult

__all__ = [
    "DEFAULT_GSM8K_PROMPT_TEMPLATE",
    "GSM8K_COT_4SHOT_PROMPT_TEMPLATE",
    "GSM8K_COT_4SHOT_TERMINAL_PROMPT_TEMPLATE",
    "GSM8K_COT_4SHOT_MULTITURN_TERMINAL_TARGET_TEMPLATE",
    "GSM8K_ZERO_SHOT_BOXED_PROMPT_TEMPLATE",
    "MATH_CONFIGS",
    "MATH_DATASET_ID",
    "MODELSCOPE_MATH_DATASET_ID",
    "MATH_ZERO_SHOT_BOXED_PROMPT_TEMPLATE",
    "build_gsm8k_cot_4shot_multiturn_terminal_messages",
    "ChatMessage",
    "STRICT_GSM8K_PROMPT_TEMPLATE",
    "DatasetExample",
    "DatasetLoadResult",
    "load_dataset_examples",
    "load_dataset_result",
    "load_gsm8k_examples",
    "load_math_examples",
]
