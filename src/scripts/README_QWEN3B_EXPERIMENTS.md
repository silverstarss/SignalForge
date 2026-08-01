# Qwen 3B GRPO Fair Experiments

Research question:

> Under a fixed rollout-token budget, can curriculum sampling based on historical pass rate and group reward variance improve training efficiency over random sampling and dynamic sampling?

This folder starts with A/B. C-E should keep the same control variables and only add one mechanism at a time.

| ID | Name | Mechanism | Status |
| --- | --- | --- | --- |
| A | GRPO baseline | random sampling, standard GRPO | `01_exp_a_grpo_baseline_qwen3b.sh` |
| B | GRPO + Dynamic Sampling | filter all-correct/all-wrong groups before update | `02_exp_b_grpo_dynamic_sampling_qwen3b.sh` |
| C | B + Clip-Higher | set `CLIP_RATIO_HIGH=0.28` or later DAPO value | planned |
| D | C + Overlong Reward Shaping | add overlong reward buffer/penalty | planned |
| E | D + Adaptive Curriculum Sampling | sample medium-difficulty items by historical pass@k and reward variance | planned |

## Fixed Controls

Keep these identical across A-E unless a run is explicitly marked as an ablation:

- Initial model: `MODEL_PATH`, default `Qwen/Qwen2.5-3B-Instruct`.
- Dataset: `data/gsm8k/train.parquet` and `data/gsm8k/test.parquet`.
- Rollout budget: `TRAIN_BATCH_SIZE * ROLLOUT_N * TOTAL_TRAINING_STEPS` raw rollouts, plus logged extra rollouts.
- Train batch: default `TRAIN_BATCH_SIZE=32`, `ROLLOUT_N=8`, `PPO_MINI_BATCH_SIZE=32`.
- Inference: train `temperature=1.0`, `top_p=1.0`; validation `VAL_ROLLOUT_N=8`, `temperature=0.7`, `top_p=0.95`.
- Validation schedule: `VAL_BEFORE_TRAIN=True`, `TEST_FREQ=25` by default.
- Checkpoint selection: choose best validation reward/pass@1 under the same validation schedule; also report final-step metrics.

## AutoDL Run

Inside the AutoDL container:

```bash
cd /workspace
bash /workspace/scripts_grpo/00_check_env.sh
bash /workspace/scripts_grpo/01_prepare_gsm8k.sh
```

Smoke A/B first:

```bash
TOTAL_TRAINING_STEPS=5 TRAIN_BATCH_SIZE=8 ROLLOUT_N=4 VAL_ROLLOUT_N=4 \
  bash /workspace/scripts/01_exp_a_grpo_baseline_qwen3b.sh

TOTAL_TRAINING_STEPS=5 TRAIN_BATCH_SIZE=8 ROLLOUT_N=4 VAL_ROLLOUT_N=4 PPO_MINI_BATCH_SIZE=8 \
  bash /workspace/scripts/02_exp_b_grpo_dynamic_sampling_qwen3b.sh
```

Full comparable A/B:

```bash
TOTAL_TRAINING_STEPS=300 SAVE_FREQ=50 TEST_FREQ=25 \
  bash /workspace/scripts/01_exp_a_grpo_baseline_qwen3b.sh

TOTAL_TRAINING_STEPS=300 SAVE_FREQ=50 TEST_FREQ=25 \
  bash /workspace/scripts/02_exp_b_grpo_dynamic_sampling_qwen3b.sh
```

A single A800 80GB should be comfortable for this 3B setup with FSDP offload and vLLM colocated rollout. If memory is tight, first reduce `ROLLOUT_GPU_MEM_UTIL=0.40`, then `TRAIN_BATCH_SIZE=16`, then `ROLLOUT_N=4` for smoke only. Do not compare reduced-budget runs against full-budget runs.

## Metrics To Report

Minimum table columns:

- `pass@1`, `pass@k` from validation dumps.
- reward mean and variance.
- response length.
- entropy, KL or PPO KL.
- clip fraction and lower clip fraction.
- all-correct/all-wrong group ratio.
- accepted group ratio.
- raw rollout count, accepted rollout count, rejected rollout count, extra rollout count.
- rollout response tokens, accepted response tokens.
- wall-clock time and GPU utilization from logs/GPU CSV.

Summarize one run:

```bash
python /workspace/scripts/summarize_qwen3b_metrics.py \
  /workspace/outputs/qwen3b_grpo_fair_gsm8k/A_grpo_baseline_qwen25_3b_gsm8k
```

For B, the trainer logs `dynamic_sampling/*`. In this first implementation `dynamic_sampling/extra_rollout_count=0`: B filters without replacement, so it uses the same raw rollout budget but fewer accepted update samples. If later you implement DAPO-style replacement sampling, increment and report extra rollouts explicitly.

## Public README Checklist

Your GitHub writeup should include:

- experiment table A-E;
- exact commands and environment;
- training configs and fixed controls;
- curves for validation pass@1/pass@k, reward variance, response length, KL, entropy, clip fraction;
- dynamic/curriculum accepted group ratio and extra rollout compute;
- GPU hours or wall-clock time;
- failed runs and OOM settings;
- conclusion and limitations;
- one core-code flowchart showing dataset -> rollout -> reward -> dynamic filter -> advantage -> PPO update -> validation.
