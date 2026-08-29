# SignalForge v2 Experiment Protocol

**Status:** FROZEN for the preregistered single-GPU formal A/B comparison (2026-08-26). Any parameter change requires a new experiment identity and protocol amendment.

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

Frozen formal A/B horizon and optimizer:

```text
training_steps = 300
seed = 42
B_t = train_batch_size = 32 prompt groups
G = 8 responses per prompt

optimizer = AdamW (torch.optim)
learning_rate = 1e-6
betas = (0.9, 0.999)
weight_decay = 0.01
lr_scheduler = constant
warmup_steps = 0
warmup_ratio = 0.0
ppo_epochs = 1
KL reward penalty = off
KL actor loss = off
```

The reviewed single-GPU execution settings are shared across A and B and are
frozen as non-algorithmic memory settings:

```text
ppo_mini_batch_size = 32 prompt groups = 256 responses
ppo_micro_batch_size_per_gpu = 1 response
actor use_dynamic_bsz = true
actor ppo_max_token_len_per_gpu = 4096
rollout/ref log_prob_micro_batch_size_per_gpu = 1
rollout/ref log_prob_max_token_len_per_gpu = 4096
rollout max_model_len = 3328
rollout max_num_batched_tokens = 4096
rollout max_num_seqs = 8
rollout gpu_memory_utilization = 0.45
Ulysses sequence parallel size = 1
```

Dynamic token batching determines the number of micro-batches required by the
actual sequence lengths; there is no separately frozen constant gradient accumulation count. The semantic optimizer batch remains exactly `B_t * G =
256` complete responses.

Frozen rollout constants for the current reproduction:

```text
G = 8
temperature = 1.0
max_prompt_length = 1792
max_response_length = 1536
```

Frozen HIVE constants for the current single-GPU formal reproduction:

```text
B_t = 32
B_cand = 48 = 3 * B_t / 2
b_raw = 32
b_min = 8
eta = 1.25
max_topup_rounds = 4
survival_epsilon = 1e-6
selector_rng_seed = 42

k_off = upper_trim_ratio = 0.25
k_keep = keep_ratio = 0.50

p_easy_init = 0.5
p_hard_init = 0.5
p_default = 0.5
alpha_total = 0.25
alpha_easy = 1/12
alpha_hard = 1/6
delta_p = 0.01
p_min = 0.05
p_max = 0.95
epsilon_p = 0.01
lambda = 1.0

prompt_entropy_micro_batch_size = 1
```

The exact `B_cand = 3 * B_t / 2` derivation, Appendix B.3 Stage-2 entropy
band, and prompt-count rounding down to a complete `G` multiple are faithful
HIVE semantics, not tunable single-GPU adaptations. Rounding continues to
remove the lowest-entropy prompts within the pre-round retained band and to
classify them separately as `rounding_dropped`.

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

HIVE preflight must reject `b_min > B_cand`. The current single-GPU formal
reproduction freezes `b_min = 8`; this paper-scale adaptation is recorded as
HIVE-006 in `docs/hive/HIVE_DEVIATIONS.md`. A different reduced value supplied
for a smoke test must still be explicitly labeled smoke-only and does not
revise the formal protocol.

For HIVE only, initial candidate accumulation and adaptive top-up may cross
dataset epoch boundaries within one optimizer step. Acquisition continues in
the configured sampler order. Stable prompt IDs must not repeat within that
optimizer step, and checkpoint/resume must preserve dataloader sampler/iterator
state, the HIVE stream epoch, and any pending pre-read rows. This is a data
lifecycle rule and does not modify the HIVE selection or top-up equations.

---

## 4. Dataset

Frozen dataset mixture for the current reproduction:

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
source-local validation, deduplication, and validation-aware decontamination.
The six validation benchmarks remain immutable: confirmed exact same-problem
(`A`) and trivial-paraphrase/same-structure (`B`) matches are removed only from
the training source pools; genuinely different (`C`) candidates remain.
DAPO duplicate rows are keyed by `extra_info.index`. The frozen DAPO parquet
has SHA-256 `534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941`
and contains 100 verified identical stable-ID cycles; its 17,917 logical rows
are materialized once. Preserve source identity rather than renumbering the
mixture:

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

The complete validated source pools contain 7,496 MATH and 17,917 DAPO
prompts. The frozen complete-source decontamination audit generated 3,620
train/benchmark candidate pairs across 264 validation rows. All 224 pairs with
token-LCS >= 0.8 or normalized-character similarity >= 0.9 were reviewed.
The pair-level classifications are 51 `A`, 69 `B`, and 3,500 `C`; after
collapsing repeated matches by stable training ID and giving `A` precedence,
45 exact and 61 near-duplicate prompts are removed. This removes 6 MATH and
100 DAPO prompts, leaving 7,490 clean MATH and 17,817 clean DAPO prompts.

Recompute the maximal exact ratio only after these removals:

```text
q = min(floor(7490 / 3), 17817) = 2496
MATH = 3q = 7488
DAPO = q = 2496
total = 9984
```

All 9,984 stable prompt IDs are unique. Of the nine overlaps reported by the
older selected-pool audit, eight are confirmed and removed; `math:7163` is
kept because the prior operator-dropping normalization conflated `+z^2` with
`-z^2`. The validation rows themselves are unchanged.

With the frozen Qwen2.5-3B-Instruct chat template and generation-prompt suffix,
the clean formal pool's untruncated prompt lengths have `p99 = 480`, `p99.9 =
953`, and `max = 1704` tokens. Therefore the formal `max_prompt_length` remains
`1792`, the next 128-token boundary above the observed maximum. No selected
prompt is removed or truncated to obtain this limit. Together with
`max_response_length = 1536`, rollout context must support at least `3328`
tokens. The auditable pool, candidate manifest, decision summary, and length
report live under
`artifacts/formal_data/hive_math75_dapo25_seed42_validation_clean_max_exact_3to1`.

Frozen training parquet SHA-256: `94c4d168cf911797a6694a6be2c4ebc3c4ae677b51c0e03b7988227e0946de5f`.


The formal launcher verifies this hash before preflight or execution. It also
rejects command-line changes to frozen semantic parameters. Only approved
preflight flags and exact-path checkpoint resume controls may pass through;
any other change requires a protocol amendment and a new experiment identity.

The pinned AMC23 and Gaokao2023en snapshots are both retained in full and are
still reported separately and in the preregistered six-benchmark average. A
prior review note stated that they contain 8 cross-benchmark duplicate
problems; operator-preserving normalized-text inspection of the pinned
snapshots finds 9. The nine benchmark-qualified pairs are recorded in
`decontamination_summary.json`; this discrepancy does not remove validation
rows or change the reporting rule.


## 4.1 Frozen Validation Suite

The formal validation suite is immutable and contains 1,902 prompts:

```text
MATH-500                         500
AIME 2024                         30
AMC23                             40
Minerva Math                     272
Gaokao2023en                     385
OlympiadBench English text-math 675
total                           1902
```

Pinned source snapshots:

```text
Qwen evaluation revision: a45202bd16f1ec06f433442dc1152d0074773465
MATH-500 revision:         6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be
```

The trainer-ready artifact is:

```text
artifacts/validation_data/qwen_math_a45202bd_math500_6e4ed1a2/formal_validation.parquet
SHA-256: cff36876612e3e55bb963e1f05a33b60c86cb7907d18befa90e38718566a4301
```

All validation prompts use the same canonical boxed-answer prompt as training.
Validation-qualified stable IDs use
`validation:<benchmark>:<benchmark_id>`; none overlap training IDs. The Qwen
chat-template token scan has `p99=433` and `max=1303`, so no validation row is
removed or truncated at `max_prompt_length=1792`.

OlympiadBench rows marked `is_multiple_answer` use the source-declared
semantics: top-level answers form an unordered set, while components inside an
individual tuple remain ordered. Gaokao answer leakage is removed according to
the audited normalization tests. All benchmarks use the shared three-state
verifier; validation accuracy is the binary `correct` field, not mean scalar
reward.

Validation decoding and schedule are frozen:

```text
n = 1
greedy / do_sample = false
temperature = 0
top_p = 1.0
top_k = -1
max_response_length = 1536
steps = 0, 50, 100, 150, 200, 250, 300
```

The primary model-selection metric is the unweighted arithmetic mean of the
six benchmark accuracies:

```text
val/six_benchmark_mean_accuracy
```

It is not the prompt-count-weighted `val/pass_at_1`. Each run performs 1,902
generations per validation point and 13,314 validation generations over the
seven scheduled evaluations. Actual generated validation tokens and timing
must be retained.

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

Do not judge HIVE only by final accuracy. The primary model-selection metric is
`val/six_benchmark_mean_accuracy`; report every individual benchmark accuracy
as well.

Report at minimum:

```text
validation accuracy vs training step
validation accuracy vs generated responses
validation accuracy vs generated response tokens
validation accuracy vs rollout wall-clock time
validation accuracy vs total wall-clock time

total generated responses
total generated response tokens
effective prompt ratio
zero-variance rollout ratio

rollout wall time
selector overhead
total wall time
```

All discarded HIVE rollouts count toward compute usage. Formal A and B use the
same checkpointed `compute/*` counters and the same validation schedule.

Checkpoint policy is frozen:

```text
save steps = 50, 100, 150, 200, 250, 300
server retention during each run = all six scheduled checkpoints
max actor/critic checkpoints to keep = 6
minimum archived artifacts after review = best checkpoint + final checkpoint
```

No scheduled checkpoint is automatically evicted during the 300-step run.
Cleanup may occur only after the run is complete, validation results have been
reviewed, and the required artifacts have been archived. `best_checkpoint.json`,
resolved config, logs, validation dumps, selector state, common compute counters,
and HIVE compute counters accompany the archived checkpoints. Resume must
preserve all three global-step values and must not duplicate committed visits.

Verifier infrastructure policy is fail-fast:

```text
process-isolated timeout = 120 seconds
verify_timeout_fallback = false
```

A parser timeout or verifier exception aborts the run; it must not be silently
converted into an extraction-failure reward of 0.0.


---

## 7. Development Gates

All listed gates have passed for the frozen configuration. Formal launch still requires a clean committed worktree and a passing strict preflight on the active GPU node.

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
