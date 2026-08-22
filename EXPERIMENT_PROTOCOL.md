# SignalForge v2 Experiment Protocol

**Status:** DRAFT until the final 3B HIVE semantic smoke and remaining development gates pass.

## 1. Goal

Evaluate whether HIVE improves **RLVR rollout efficiency** relative to vanilla GRPO while maintaining comparable reasoning performance.

This is a HIVE reproduction project, not a search over new algorithms.

---

## 2. Experiments

### A — GRPO Baseline

Standard veRL GRPO with fixed group size:

```yaml
group_size: 8
```

No prompt pre-filtering or HIVE selection.

### B — HIVE

Same training setup as A, with HIVE prompt selection enabled according to:

```text
docs/hive/HIVE_IMPLEMENTATION_SPEC.md
```

The GRPO objective itself should remain unchanged.

---

## 3. Fixed Across A/B

Unless explicitly documented otherwise:

```text
model
dataset
dataset split
prompt template
reward/verifier
G = 8
sampling temperature
max prompt length
max response length
optimizer
learning rate
effective training batch definition
validation set
validation decoding
training horizon / stopping rule
random seed policy
```

Target model:

```text
Qwen2.5-3B-Instruct
```

Frozen rollout constants for the current reproduction:

```text
G = 8
temperature = 1.0
max_response_length = 1536
```

Target hardware:

```text
1 × RTX PRO 6000D 84GB
```

Paper-scale batch sizes must be reduced as necessary for single-GPU training, but A and B must use the same effective training configuration.

For HIVE adaptive top-up, the paper default is `b_min = 64`, and the
deployed paper configurations satisfy:

```text
B_cand = 3 * B_t / 2
b_min <= B_cand
```

HIVE preflight must reject `b_min > B_cand`. Reduced single-GPU runs may
use a smaller `b_min`. A reduced value supplied for a smoke test must be
explicitly labeled smoke-only and does not freeze the formal protocol.
Any formal reproduction value other than `64` must be recorded in
`docs/hive/HIVE_DEVIATIONS.md`.

---

## 4. Dataset

Frozen dataset candidate for the current reproduction:

```text
75% MATH + 25% DAPO-Math
```

This is a Qwen2.5-3B-Instruct model-relative dataset adaptation selected from
the reviewed `G=8`, `temperature=1.0`, `max_response_length=1536`
calibration. It is not presented as a paper-default dataset mixture.

Sources and revisions:

```text
MATH: EleutherAI/hendrycks_math
revision: 21a5633873b6a120296cce3e2df9d5550074f4a3
split: train, all seven subject configurations

DAPO: BytedTsinghua-SIA/DAPO-Math-17k
revision: 65877096c24ffa7abc4e4fa5edb95cf3413a5674
split: train
```

Construct the prompt-level pool at an exact `3:1` MATH:DAPO ratio after
source-local validation and deduplication. DAPO duplicate rows are keyed by
`extra_info.index`; conflicting duplicates are rejected. Preserve source
identity rather than renumbering the mixture:

```text
math:<source_row_id>
dapo:<extra_info.index>
```

Both sources use exactly one canonical prompt:

```text
Solve the following math problem step by step.
Put your final answer in \boxed{...}.

{problem}
```

DAPO's original `Answer: ...` requirement is removed before applying this
template. Its bare gold answer is wrapped and validated against the same boxed
LaTeX verifier used for MATH. The reviewed calibration raw results and source
statistics live under
`artifacts/calibration/hive_dataset/source_pools_math256_dapo256_seed42_r1536`.
Do not select or revise the mixture using effective ratio alone.

---

## 5. Reward

Use one shared math verifier/reward implementation across A and B.

The reward semantics are frozen as:

```text
correct                   -> 1.0
extractable but incorrect -> 0.1
extraction failure        -> 0.0
```

Zero-variance reward groups are classified exactly as:

```text
easy zero-var  = all rewards == 1.0
hard zero-var  = all rewards == 0.1
other zero-var = any other constant-reward group, including all 0.0
```

The same semantics and classification must be used across A and B.

---

## 6. Primary Evaluation

Do not judge HIVE only by final accuracy.

Report at minimum:

```text
validation accuracy vs training step
validation accuracy vs generated responses
validation accuracy vs generated response tokens

total generated responses
total generated response tokens
effective prompt ratio
zero-variance rollout ratio

rollout wall time
selector overhead
total wall time
```

Save and report both:

```text
best checkpoint
final checkpoint
```

All discarded HIVE rollouts still count toward compute usage.

---

## 7. Development Gates

Formal experiments may begin only after:

```text
1. old GRPO pipeline runs on the RTX 6000D;
2. Qwen2.5-3B GRPO memory smoke passes;
3. dataset calibration is complete;
4. dataset and reward semantics are frozen;
5. HIVE unit tests pass;
6. 2–5 step HIVE smoke passes;
7. checkpoint/resume passes;
8. 50–100 step mechanism pilot behaves sensibly;
9. rollout/token accounting is verified.
```

Smoke tests establish correctness, not performance.

---

## 8. Reproducibility

Every formal run must record:

```text
git commit
resolved Hydra config
Python / PyTorch / CUDA / veRL / vLLM versions
model revision
dataset version
seed
GPU type
generated responses
generated response tokens
wall-clock statistics
```

Any deliberate deviation from the HIVE paper must be recorded in:

```text
docs/hive/HIVE_DEVIATIONS.md
```
