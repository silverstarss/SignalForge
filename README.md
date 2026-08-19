# SignalForge

SignalForge is an LLM post-training project built on **veRL + vLLM** for studying rollout efficiency in RLVR.

## Current Version: SignalForge v2

SignalForge v2 focuses on reproducing **HIVE (History-Informed Verification of the Learning Edge)**.

The core question is:

> Can expensive rollout compute be concentrated on prompts near the model's current learning edge without sacrificing reasoning performance?

HIVE addresses this with:

1. **history-informed prompt filtering**;
2. **current-policy prompt-entropy verification**;
3. standard `G=8` rollouts on selected prompts;
4. post-rollout zero-variance filtering and adaptive top-up.

Detailed algorithmic behavior is specified in:

```text
docs/hive/HIVE_IMPLEMENTATION_SPEC.md
```

## Target Setup

```text
Framework: veRL + vLLM
Model: Qwen2.5-3B
Task: mathematical reasoning / RLVR
Hardware target: 1 × RTX PRO 6000D 84GB
Group size: G=8
```

The exact training dataset and single-GPU batch configuration are frozen separately in `EXPERIMENT_PROTOCOL.md`.

## Experiments

### A — GRPO

Vanilla GRPO serves as the routine baseline.

### B — HIVE

GRPO with faithful HIVE prompt selection and rollout-efficiency accounting.

Primary comparisons focus on both reasoning performance and compute efficiency, including:

```text
validation accuracy
generated responses
generated response tokens
effective prompt ratio
zero-variance rollout ratio
rollout wall time
total wall time
```

## Repository Guidance

For development instructions:

```text
AGENTS.md
```

For experiment constraints:

```text
EXPERIMENT_PROTOCOL.md
```

For the HIVE implementation specification:

```text
docs/hive/HIVE_IMPLEMENTATION_SPEC.md
```

SignalForge v1 documents and results should be retained under `docs/v1/` for historical reference.
