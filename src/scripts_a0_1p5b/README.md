# Qwen2.5-1.5B Experiment A Entry Plan

This directory is reserved for 1.5B launch scripts and run notes. It must not
fork the trainer, reward adapter, dataset format, local veRL source, or the
validated 0.5B regression path in `src/scripts_a0/08_run_a0_0p5b_regression.sh`
and `src/scripts_a0/09_reload_a0_0p5b_regression_checkpoint.sh`.

## Frozen Stage 0 Inputs

Use the frozen Signal Forge v1 data:

```text
data/processed/signal_forge_v1/train.parquet
data/processed/signal_forge_v1/validation_id.parquet
data/processed/signal_forge_v1/test_id.parquet
data/processed/signal_forge_v1/manifest.json
```

Training uses `train.parquet`. Checkpoint selection uses `validation_id.parquet`.
`test_id.parquet` is reserved for final evaluation after checkpoint selection.

Keep these scientific settings fixed for A:

```text
model: Qwen2.5-1.5B-Instruct
algorithm: standard GRPO
rollout.n: 8
max_response_length: 768
reward: boxed-only binary Math-Verify correctness
format reward: disabled
length reward: disabled
dynamic sampling: disabled
clip-higher: disabled
curriculum: disabled
```

## Planned Run Order

1. `01_preflight_1p5b_a800.sh`
   Run fast and deep preflight only. This checks GPU/CUDA, model/tokenizer,
   dataset, reward, batch invariants, disk, environment, and local veRL source.

2. `02_run_1p5b_short_smoke.sh`
   Run 2-3 optimizer steps. This validates Ray, vLLM, rollout, reward,
   advantage, actor update, validation, and checkpoint save. Do not interpret
   accuracy movement.

3. `03_run_1p5b_regression_40step.sh`
   Run 40 optimizer steps. This is still an engineering regression, not the
   formal A result. Monitor memory trend, response length, hit-max rate,
   extraction/format failure, reward, KL, entropy, clip fraction, grad norm,
   reward latency, step time, checkpoint size, disk growth, and W&B logging.

4. `04_reload_1p5b_checkpoint.sh`
   Start from the 40-step checkpoint after the original process has fully
   stopped, then continue 3-5 optimizer steps. Confirm `global_step`, optimizer
   state, LR scheduler, model weights, W&B resume, and subsequent checkpoint
   saving.

5. Formal A run
   The exact compute budget is TBD until after the 40-step regression. Freeze
   `max_optimizer_steps`, `max_generated_responses`,
   `target_rollout_response_tokens`, and GPU-hour upper bound before launching.

## Checkpoint Selection

Select best checkpoint using `validation_id` 60/40 weighted pass@1:

```text
0.60 * validation_gsm8k_pass_at_1 + 0.40 * validation_math_level_3_pass_at_1
```

Also report GSM8K pass@1, MATH Level 3 pass@1, and macro-average pass@1. Never
use `test_id` to choose checkpoints or change the formal compute budget.

## RTX6000D Migration Smoke

Use this only to validate the migrated machine/runtime before resuming formal
Experiment B work. It uses Qwen2.5-1.5B-Instruct plus boxed GSM8K only, so it is
not formal A or B evidence.

```bash
# First no-GPU/static check; skips model prefetch and does not train.
VENV_DIR=/root/miniconda3/envs/verl \
  bash src/scripts_a0_1p5b/08_run_1p5b_gsm8k_migration_smoke.sh --preflight-only

# On a GPU instance, download/reuse the model cache, download/reuse GSM8K, then run 2 steps.
VENV_DIR=/root/miniconda3/envs/verl \
  bash src/scripts_a0_1p5b/08_run_1p5b_gsm8k_migration_smoke.sh
```

Runtime locations come from `config/signal_forge.env` and can be overridden with
`MODEL_ROOT`, `DATA_ROOT`, `CACHE_ROOT`, `OUTPUT_ROOT`, `CHECKPOINT_ROOT`,
`WANDB_ROOT`, `WANDB_MODE`, and `ENABLE_WANDB`.
