# AGENTS.md

## Project Goal

SignalForge v2 is a **faithful HIVE reproduction in veRL**, using Qwen2.5-3B for math RLVR.

The primary goal is to reproduce and evaluate HIVE's rollout-selection mechanism, not to invent a new algorithm.

## Required Reading

Before any HIVE-related work, read:

1. `EXPERIMENT_PROTOCOL.md`
2. `docs/hive/HIVE_IMPLEMENTATION_SPEC.md`

`HIVE_IMPLEMENTATION_SPEC.md` is the algorithmic source of truth.

## Rules

* Inspect the existing execution path before modifying code.
* Reuse existing SignalForge / veRL infrastructure whenever possible.
* Keep GRPO baseline behavior unchanged except for necessary environment/config migration.
* Do not introduce SARA, VIGOR, replay, retention, adaptive group size, or other unrelated mechanisms.
* Do not silently change HIVE equations, thresholds, reward semantics, or budget accounting.
* Record deliberate paper deviations in `docs/hive/HIVE_DEVIATIONS.md`.
* Preserve complete prompt groups (`G=8`) throughout rollout and training.
* Count all generated rollouts/tokens, including discarded groups.
* Do not hard-code machine-specific model, dataset, checkpoint, or output paths.
* Add tests before integrating new HIVE components into the full trainer.
* Do not launch formal experiments until the protocol is frozen and smoke/pilot tests pass.

## Development Workflow

For substantial changes:

1. inspect;
2. propose the minimum architecture/code changes;
3. review;
4. implement;
5. test;
6. smoke test;
7. commit.

Do not combine unrelated refactors with HIVE implementation work.

## Current Scope

Primary experiments:

* **A: vanilla GRPO baseline**
* **B: HIVE**

SignalForge v1 experiments are historical work and must not silently influence v2 behavior.
