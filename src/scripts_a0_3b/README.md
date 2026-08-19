# Qwen2.5-3B Runtime Smoke Scripts

These scripts are for RTX6000D migration and memory validation only. They reuse
the existing veRL trainer, SignalForge Math-Verify reward adapter, and boxed
GSM8K runtime dataset. They are not formal Experiment A/B evidence.

```bash
# Static/no-GPU preflight. Does not download the model and does not train.
VENV_DIR=/root/miniconda3/envs/verl \
  bash src/scripts_a0_3b/01_run_3b_gsm8k_grpo_memory_smoke.sh --preflight-only

# GPU run. Downloads/reuses Qwen2.5-3B-Instruct and GSM8K, then runs 5 steps.
VENV_DIR=/root/miniconda3/envs/verl \
  bash src/scripts_a0_3b/01_run_3b_gsm8k_grpo_memory_smoke.sh

# If the 5-step smoke is clean, extend to 10 steps without changing code.
TOTAL_TRAINING_STEPS=10 VENV_DIR=/root/miniconda3/envs/verl \
  bash src/scripts_a0_3b/01_run_3b_gsm8k_grpo_memory_smoke.sh
```

Default memory-smoke shape:

```text
model: Qwen/Qwen2.5-3B-Instruct, cached under ${MODEL_ROOT}/Qwen when prefetched
algorithm: vanilla GRPO
data: boxed GSM8K only
rollout.n: 8
train prompts/step: 1
max_prompt_length: 512
max_response_length: 768
steps: 5 by default, override TOTAL_TRAINING_STEPS=10
dynamic sampling: disabled
W&B: disabled/offline unless ENABLE_WANDB=1
```
