# Experiment A0 Smoke Test

A0 is the minimum end-to-end check for Experiment A. It is not an accuracy run.
It keeps the frozen A protocol shape: Qwen2.5-1.5B-Instruct, standard GRPO,
GSM8K + MATH Level 3, `n=8`, `max_response_length=768`, boxed-only binary
Math-Verify reward, no Dynamic Sampling, no Clip-Higher, no length shaping, and
no curriculum.

## Expected Cloud Layout

The scripts auto-detect the local WSL layout, but the clean AutoDL layout is:

```text
/workspace/verl
/workspace/src/RewardScope/src
/workspace/src
/workspace/data
```

`/workspace/src` is this git repository. RewardScope is used only as an
importable frozen verifier package; the training process does not import its
samplers, runners, reports, or plotting code.

## Run Order

```bash
bash /workspace/src/scripts_a0/00_prepare_a0_data.sh
bash /workspace/src/scripts_a0/01_check_reward_equivalence.sh
bash /workspace/src/scripts_a0/02_check_verl_reward_manager.sh
bash /workspace/src/scripts_a0/03_run_a0_grpo_smoke.sh
bash /workspace/src/scripts_a0/04_reload_a0_checkpoint.sh
```

If AutoDL cannot access the MATH test dataset during data export, provide an
independent held-out inputs file:

```bash
bash /workspace/src/scripts_a0/00_prepare_a0_data.sh \
  --math-val-inputs /workspace/src/RewardScope/outputs/<held-out-math-level-3>/inputs.jsonl
```

## What A0 Must Prove

- parquet rows preserve `data_source`, chat `prompt`, `reward_model.ground_truth`, and `extra_info`;
- train and validation prompt IDs do not overlap;
- gold answers are parseable by the frozen RewardScope Math-Verify verifier;
- RewardScope verifier and veRL adapter agree 100% on saved rollouts;
- veRL reward manager passes response-only text to `compute_score`;
- reward is written at the final valid response token;
- reward extra fields survive through the reward manager;
- `trainer.use_v1=False` avoids the local `transfer_queue` import failure;
- checkpoint save and reload both work.

## Budget Notes

A0 defaults to `TRAIN_BATCH_SIZE=5`, `ROLLOUT_N=8`, and
`TOTAL_TRAINING_STEPS=3`, so it requests 120 training rollouts before
validation. The run config log also records an upper-bound response rollout
token budget: `TRAIN_BATCH_SIZE * ROLLOUT_N * TOTAL_TRAINING_STEPS * 768`.
Formal A should scale data and steps, but keep the same reward, prompt,
validation files, rollout settings, and checkpoint selection rule used for the
later B-E controlled ablations.

## Local Data And Models

MATH validation is local-only. The default path is:

```text
/workspace/src/RewardScope/raw/competition_math/test
```

The copied raw test set comes from ModelScope `opencompass___competition_math`
and contains the original `problem`, `level`, `type`, and `solution` JSON files.
`prepare_a0_data.py` filters `Level 3` and extracts the final boxed gold answer
with the vendored RewardScope verifier.

For local Qwen weights, mount your host cache into the container:

```bash
-v /home/tutu/tinyvr/models/Qwen:/workspace/models/Qwen:ro
```

A0 auto-detects these model directories before falling back to the HuggingFace id:

```text
/workspace/models/Qwen/Qwen2.5-1.5B-Instruct
/workspace/models/Qwen/Qwen2___5-1___5B-Instruct
/home/tutu/tinyvr/models/Qwen/Qwen2.5-1.5B-Instruct
/home/tutu/tinyvr/models/Qwen/Qwen2___5-1___5B-Instruct
```

You can always override explicitly:

```bash
TOKENIZER_PATH=/workspace/models/Qwen/Qwen2___5-1___5B-Instruct MODEL_PATH=/workspace/models/Qwen/Qwen2___5-1___5B-Instruct bash /workspace/src/scripts_a0/03_run_a0_grpo_smoke.sh
```

## Offline Math-Verify

The repository vendors the pure-Python Math-Verify packages needed by RewardScope's verifier:

```text
/workspace/src/vendor_python
```

A0 scripts prepend this directory to `PYTHONPATH`, so the normal path does not
need `pip install -e ".[math]"` or network access for Math-Verify. To verify it
manually:

```bash
export PYTHONPATH=/workspace/src:/workspace/src/vendor_python:/workspace/src/RewardScope/src:/workspace/verl:$PYTHONPATH
python - <<'PY'
from math_verify import parse, verify
from rewardscope.verification.math_verify import MathVerifyNumericVerifier
print("math_verify offline ok")
print(MathVerifyNumericVerifier(mode="training").verify(r"\boxed{2}", "2"))
PY
```

## 0.5B Regression On 4090

This is an engineering regression test for the A0/A chain. It changes the model
size only; it keeps `n=8`, `max_response_length=768`, boxed-only Math-Verify,
standard GRPO, and disables Dynamic Sampling, Clip-Higher, overlong shaping, and
curriculum.

If the 0.5B model tarball is mounted at `/workspace/qwen25_0p5b_instruct.tar.gz`,
prepare the local model directory first:

```bash
bash /workspace/src/scripts_a0/05_unpack_qwen25_0p5b_model.sh
```

Prepare a slightly larger local regression dataset:

```bash
bash /workspace/src/scripts_a0/06_prepare_a0_0p5b_regression_data.sh
```

Run the very short chain check:

```bash
bash /workspace/src/scripts_a0/07_run_a0_0p5b_short.sh
```

Run the medium regression check:

```bash
bash /workspace/src/scripts_a0/08_run_a0_0p5b_regression.sh
```

Reload the medium regression checkpoint:

```bash
bash /workspace/src/scripts_a0/09_reload_a0_0p5b_regression_checkpoint.sh
```

The medium default is 50 optimizer steps, `TRAIN_BATCH_SIZE=5`, `ROLLOUT_N=8`,
deterministic validation `VAL_ROLLOUT_N=1`, and `MAX_RESPONSE_LENGTH=768`. You can lower only step count,
subset size, or micro-batches for debugging; do not lower group size, response
length, prompt template, reward semantics, or GRPO path when using it as the A0
regression gate.


## Online Observability

A0 now logs lightweight online metrics that are not meant to replace RewardScope
offline analysis:

- `reward/*` for raw correctness, extraction and format ratios;
- `group/*` for all-correct/all-wrong/mixed prompt groups based on raw correctness;
- `budget/*` for generated rollout tokens and cumulative fair-comparison budget;
- `length/*` for response percentiles and correct/incorrect length means;
- `val/*` for deterministic validation aliases and best-checkpoint metadata.

Validation defaults are deterministic (`n=1`, `temperature=0`, `do_sample=False`).
Training rollouts keep the A protocol shape (`n=8`, temperature 1.0, max response
length 768). Bounded JSONL dumps are controlled by `ROLLOUT_DUMP_INTERVAL`,
`ROLLOUT_DUMP_MAX_RECORDS`, and `VALIDATION_DUMP_MAX_RECORDS`.
