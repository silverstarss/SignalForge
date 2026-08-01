# RewardScope

RewardScope is a lightweight diagnostics tool for RLVR/GRPO preparation. It is
used before training to measure dataset difficulty, extraction reliability,
reward signal, and token cost for GSM8K and MATH prompts.

The project is now feature-frozen. Future work should be limited to bug fixes,
documentation, and necessary compatibility updates.

## What It Does

- Loads GSM8K and MATH examples.
- Builds boxed final-answer prompts.
- Samples decoder-only Transformers models.
- Verifies answers with Math-Verify in evaluation or training mode.
- Persists rollout JSONL and reproducibility metadata.
- Computes prompt-group metrics: all-wrong, mixed, all-correct, pass@k, reward
  variance, extraction failure, hit-max, and token efficiency.
- Writes analysis reports and optional plots.

## Minimal Usage

Run an experiment from a YAML config through the Python API:

```bash
.venv/bin/python - <<'PY'
from rewardscope.runners import run_experiment_from_yaml

artifacts = run_experiment_from_yaml("configs/math-grpo-train-level-3-64-max768.yaml")
print(artifacts.output_dir)
print(artifacts.summary)
PY
```

Use a config in `configs/` as the source of truth. The current recommended
microscope configs are:

- `configs/gsm8k-grpo-train-zero-shot-boxed-128.yaml`
- `configs/math-grpo-train-level-1-2-128.yaml`
- `configs/math-grpo-train-level-3-128.yaml`
- `configs/math-grpo-train-level-3-64-max768.yaml`

## Verification Policy

- GSM8K numeric experiments can use Math-Verify expression parsing.
- MATH experiments must use `math_verify_latex` with training mode.
- Training mode is boxed-only so RewardScope measures the reward signal that
  downstream GRPO training would actually see.

Do not change prompt text, verifier semantics, metric definitions, JSONL fields,
or artifact filenames without an explicit migration plan.

## Checks

```bash
.venv/bin/python -m pytest
git diff --check
```

There are no configured lint or type-check commands beyond the test suite and
whitespace validation.
