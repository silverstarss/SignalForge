# Qwen 3B GRPO Dynamic Sampling on veRL

This repo contains scripts and notes for GSM8K GRPO experiments on veRL.

Experiments:
- A: GRPO baseline
- B: GRPO + Dynamic Sampling

The veRL source change is stored as a patch:

- patches/verl_dynamic_sampling_v0.patch

Base veRL commit:

dcb222b3c5f64c8280088856fb7f2c1c7c5706ec

See scripts/README_QWEN3B_EXPERIMENTS.md for commands, metrics, and fairness controls.
