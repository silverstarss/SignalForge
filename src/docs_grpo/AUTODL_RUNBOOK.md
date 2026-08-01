# AutoDL GRPO Runbook

This runbook is for the first reproducible veRL GRPO baseline:

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- dataset: GSM8K only
- algorithm: GRPO
- rollout backend: vLLM
- target metrics: reward, accuracy, response length, entropy, KL-related logs, throughput/timing, GPU memory, before/after cases

## 0. What These Scripts Assume

The scripts follow veRL official patterns:

- Docker image from the veRL DockerHub examples: `verlai/verl:vllm011.latest`
- GSM8K preprocessing via `examples/data_preprocess/gsm8k.py --local_save_dir`
- training entrypoint: `python -m verl.trainer.main_ppo`
- GRPO setting: `algorithm.adv_estimator=grpo`
- vLLM rollout setting: `actor_rollout_ref.rollout.name=vllm`

For this checkout, `trainer.use_v1=False` is intentional. The V1 trainer imports `transfer_queue`, which is not available in the current Docker image.

## 1. Local Container

Start the container with an explicit bash entrypoint. Without this, the vLLM image may start its default vLLM server.

```bash
cd ~/grpo
HOST_IP=$(ip route | grep default | awk '{print $3}')

sudo docker run -it \
  --name verl \
  --entrypoint /bin/bash \
  --runtime=nvidia \
  --gpus all \
  --shm-size=16g \
  --network host \
  --cap-add=SYS_ADMIN \
  --env HTTP_PROXY=http://${HOST_IP}:6789 \
  --env HTTPS_PROXY=http://${HOST_IP}:6789 \
  -v ~/grpo:/workspace \
  -w /workspace \
  verlai/verl:vllm011.latest
```

Inside the container:

```bash
cd /workspace
python3 -m venv --system-site-packages .venv-vllm
source /workspace/.venv-vllm/bin/activate
cd /workspace/verl
python -m pip install --no-deps -e .
```

Do not blindly upgrade `setuptools` in this image. vLLM 0.11 requires `setuptools>=77,<80`.

## 2. Check Environment

```bash
bash /workspace/scripts_grpo/00_check_env.sh
```

Expected:

- `torch ok`
- `vllm ok`
- `ray ok`
- `verl ok`
- `nvidia-smi` shows the GPU

## 3. Prepare GSM8K

```bash
bash /workspace/scripts_grpo/01_prepare_gsm8k.sh
```

Expected files:

```text
/workspace/data/gsm8k/train.parquet
/workspace/data/gsm8k/test.parquet
```

## 4. Save Before-Training Cases

```bash
bash /workspace/scripts_grpo/05_sample_before_cases.sh
```

Output:

```text
/workspace/outputs/cases/qwen25_1p5b_before.jsonl
```

This records strict accuracy, flexible accuracy, format errors, response length, and latency for a small fixed case set.

## 5. Local Smoke Test

Local WSL + 12GB GPU may still fail because of vLLM/CUDA compatibility. Treat this as optional.

```bash
bash /workspace/scripts_grpo/02_local_grpo_tiny_smoke.sh
```

If it fails with vLLM CUDA errors on WSL, stop here locally and move to AutoDL. The script is deliberately tiny and is not an experiment result.

## 6. Package for AutoDL

Run on the WSL host, not inside Docker:

```bash
cd ~/grpo
bash scripts_grpo/06_pack_for_autodl.sh
```

Upload the generated `grpo_verl_autodl_*.tar.gz` to AutoDL.

## 7. AutoDL Setup

On AutoDL, choose a GPU in this order:

- comfortable: 48GB GPU such as A40, A6000, L40
- cheap first pass: 24GB 4090

Use the same Docker image:

```bash
docker run -it \
  --name verl \
  --entrypoint /bin/bash \
  --runtime=nvidia \
  --gpus all \
  --shm-size=32g \
  --network host \
  --cap-add=SYS_ADMIN \
  -v /root/grpo:/workspace \
  -w /workspace \
  verlai/verl:vllm011.latest
```

Inside the container:

```bash
cd /workspace
python3 -m venv --system-site-packages .venv-vllm
source /workspace/.venv-vllm/bin/activate
cd /workspace/verl
python -m pip install --no-deps -e .
```

Then run:

```bash
bash /workspace/scripts_grpo/00_check_env.sh
bash /workspace/scripts_grpo/01_prepare_gsm8k.sh
```

## 8. AutoDL Smoke Test

Run a cheap smoke test before the 1.5B experiment:

```bash
bash /workspace/scripts_grpo/03_cloud_grpo_smoke_0p5b.sh
```

This should finish quickly and produce:

```text
/workspace/outputs/cloud_grpo_smoke/qwen25_0p5b_gsm8k_smoke/logs/train_*.log
/workspace/outputs/cloud_grpo_smoke/qwen25_0p5b_gsm8k_smoke/logs/gpu.csv
```

## 9. Main 1.5B GRPO Baseline

First run a small 20-step version:

```bash
TOTAL_TRAINING_STEPS=20 ROLLOUT_N=4 bash /workspace/scripts_grpo/04_cloud_grpo_qwen25_1p5b_gsm8k.sh
```

If that succeeds, run the baseline:

```bash
TOTAL_TRAINING_STEPS=200 ROLLOUT_N=8 bash /workspace/scripts_grpo/04_cloud_grpo_qwen25_1p5b_gsm8k.sh
```

Later, if cost is acceptable:

```bash
TOTAL_TRAINING_STEPS=700 ROLLOUT_N=8 SAVE_FREQ=50 TEST_FREQ=25 bash /workspace/scripts_grpo/04_cloud_grpo_qwen25_1p5b_gsm8k.sh
```

## 10. What to Save

Always download or keep:

- `logs/train_*.log`
- `logs/gpu.csv`
- `rollout_data/`
- `validation_data/`
- `checkpoints/`
- `outputs/cases/*before*.jsonl`
- after-training case files, once an HF-compatible model path is available

The training log should contain the veRL console metrics for reward, validation score, sequence lengths, entropy/loss terms, KL-related values when enabled/reported, timing, and throughput-style performance fields. `gpu.csv` records memory separately.

## 11. If 24GB OOMs

Retry with smaller settings:

```bash
TRAIN_BATCH_SIZE=16 \
PPO_MINI_BATCH_SIZE=4 \
ROLLOUT_N=4 \
ROLLOUT_GPU_MEM_UTIL=0.40 \
ROLLOUT_MAX_NUM_BATCHED_TOKENS=2048 \
ROLLOUT_MAX_NUM_SEQS=32 \
TOTAL_TRAINING_STEPS=20 \
bash /workspace/scripts_grpo/04_cloud_grpo_qwen25_1p5b_gsm8k.sh
```

If this works, increase one variable at a time.

## 12. Interpretation

For the first real report, do not add Countdown yet. GSM8K is enough for `Qwen2.5-1.5B-Instruct` because the microscope result showed many mixed groups and only a small all-correct fraction. Track format errors separately instead of making them the main reward-shaping problem in the first run.

