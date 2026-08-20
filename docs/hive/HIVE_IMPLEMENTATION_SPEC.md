# HIVE Implementation Specification

**Project:** SignalForge v2
**Target method:** HIVE — History-Informed Verification of the Learning Edge
**Reference paper:** *Train at the Moving Edge: Efficient RL for Large Reasoning Models via Rollout Selection*, arXiv:2603.25184v2
**Target framework:** veRL + vLLM
**Target model:** Qwen2.5-3B
**Target hardware:** 1 × RTX PRO 6000D 84GB
**Status:** DRAFT — implementation specification before code modification

---

# 0. Purpose of This Document

This document is the algorithmic source of truth for implementing HIVE in SignalForge.

The goal is **not** to invent a new rollout-allocation algorithm.

The goal is:

> reproduce the deployed HIVE mechanism as faithfully as practical inside the existing SignalForge / veRL training pipeline, then evaluate its rollout efficiency against controlled GRPO / Dynamic Sampling baselines.

Codex MUST read this document before modifying HIVE-related training logic.

If the existing repository architecture conflicts with this specification, Codex must report the conflict before changing algorithmic behavior.

If this document conflicts with the HIVE paper, the paper takes precedence unless the conflict is explicitly listed in **Section 15 — Paper Ambiguities and Resolutions**.

---

# 1. Scope

## 1.1 Primary implementation target

Implement the complete HIVE selection pipeline:

```text
raw prompt stream
        |
        v
persistent prompt history
        |
        v
Stage 1:
history-informed stochastic filtering
        |
        v
Stage 2:
current-policy prompt-entropy verification
        |
        v
selected candidate prompts
        |
        v
G=8 response rollouts
        |
        v
reward computation
        |
        v
zero-variance group rejection
        |
        v
adaptive candidate top-up if necessary
        |
        v
exact fixed-size effective GRPO batch
        |
        v
policy update
        |
        v
update prompt history + selector state
```

HIVE changes **which prompts receive expensive rollouts**.

It should not unnecessarily modify the GRPO objective itself.

---

# 2. Non-Goals

Do NOT introduce the following unless a later experiment protocol explicitly requests them:

* SARA early stopping
* VIGOR progressive rollout allocation
* replay / retention
* RLEP
* curriculum learning outside HIVE
* partial rollout termination
* adaptive group size
* custom advantage estimators
* new GRPO objective variants
* KL regularization unless the protocol explicitly re-enables it
* heuristic dataset filtering unrelated to HIVE
* auxiliary difficulty models

Do not mix methods simply because they appear useful.

The first goal is a clean HIVE reproduction.

---

# 3. Conceptual Motivation

In GRPO, each prompt generates a group of (G) responses.

For rewards

[
r_1,\ldots,r_G,
]

the group-normalized advantage is based on

[
\hat A_i =
\frac{r_i-\operatorname{mean}(r)}
{\operatorname{std}(r)}.
]

If

[
\operatorname{Var}(r)=0,
]

the prompt supplies no useful relative reward signal.

Examples:

```text
1 1 1 1 1 1 1 1
```

or

```text
0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
```

produce zero reward variance.

Dynamic Sampling detects this only **after all G responses have already been generated**.

HIVE attempts to avoid part of this wasted generation cost **before rollout**.

Its central hypothesis is that useful prompts lie near a moving **learning edge**:

```text
not already mastered
+
not completely intractable
+
still uncertain under the current policy
```

Because policy competence changes during training, historical difficulty information becomes stale.

Therefore HIVE uses:

```text
Stage 1 = cheap historical prior
Stage 2 = current-policy verification
```

before spending rollout compute.

Paper reference: Section 2, Section 3.

---

# 4. Required Global Invariants

These invariants MUST hold unless `EXPERIMENT_PROTOCOL.md` explicitly overrides them.

## 4.1 Group integrity

One prompt corresponds to exactly (G) rollout responses.

Default:

```yaml
group_size: 8
```

A prompt group must never be partially passed into a GRPO update.

Do not slice individual responses in a way that produces incomplete groups.

---

## 4.2 Fixed effective training batch

Each policy update must consume exactly:

[
B_t
]

effective prompt groups, equivalent to:

[
B_t \times G
]

responses.

HIVE may inspect and rollout more candidate prompts to obtain these effective groups.

Therefore distinguish clearly between:

```text
raw prompts inspected
Stage-1 accepted prompts
Stage-2 accepted prompts
rolled-out prompts
effective prompts
training prompts
generated responses
generated response tokens
```

These quantities are NOT interchangeable.

---

## 4.3 Current-policy verification

Stage 2 entropy MUST be computed using the policy corresponding to the current training step.

Do not silently use:

* stale actor parameters;
* an outdated vLLM weight snapshot;
* reference-model weights;
* checkpoint weights from the previous evaluation step.

If Stage 2 uses the vLLM model, weight synchronization must be verified.

If that is difficult or ambiguous, prefer computing Stage 2 entropy directly with the current actor model.

---

## 4.4 Stable prompt identity

Every training prompt MUST have a stable `prompt_id`.

The ID must survive:

* dataloader reshuffling;
* multiple epochs;
* checkpoint resume;
* dataset concatenation;
* candidate rejection;
* top-up loops.

Recommended scheme:

```text
<dataset_source>:<original_row_id>
```

Examples:

```text
math:3812
dapo:12471
```

Do NOT identify prompts by their position in the current batch.

Do NOT use Python object identity.

A content hash may be stored as an additional integrity check.

---

# 5. Persistent HIVE State

Implement a persistent selector state independent of individual batches.

Suggested conceptual structure:

```python
HiveSelectorState:
    prompt_history: dict[prompt_id, PromptHistory]

    p_easy: float
    p_hard: float

    global_step: int
    selector_rng_state
```

Each prompt should store enough information to reproduce the paper's historical trace:

```python
PromptHistory:
    visits: list[PromptVisit]
```

Each visit should contain at least:

```python
PromptVisit:
    step: int

    rewards: list[float]       # length G
    reward_variance: float

    zero_variance: bool
    zero_variance_type:
        "easy"
        "hard"
        "other"
        None

    response_entropy: float | None
```

The paper describes the history trace as unbounded.

Therefore the faithful initial implementation should preserve all visits.

A later memory-optimized implementation may compress old history only after proving equivalence for quantities used by HIVE.

---

# 6. Stage 1 — History-Informed Coarse Narrowing

Paper reference: Section 3.2, Equations (3)–(6), Appendix B.3.

Stage 1 is intentionally cheap.

It MUST NOT trigger new response generation.

---

# 6.1 Zero-variance indicator

For visit (j) of prompt (i):

[
\zeta_{i,j}
===========

\mathbf 1
\left[
\operatorname{Var}(R_{i,j})=0
\right].
]

where

[
R_{i,j}
=======

{r_{i,j}^{(1)},\ldots,r_{i,j}^{(G)}}.
]

Use the actual reward group stored from the previous rollout.

Because rewards are discrete in the intended math setting, exact zero variance should normally be safe.

Nevertheless, implement the variance classification in one centralized function.

Do not scatter independent implementations across the repository.

---

# 6.2 Trailing zero-variance run

For prompt (i), define

[
z_i^{(t)}
]

as the number of consecutive zero-variance visits at the end of its history.

Examples:

```text
NZ, Z, Z, Z
→ z = 3
```

```text
Z, Z, NZ, Z
→ z = 1
```

```text
NZ
→ z = 0
```

```text
unseen prompt
→ z = 0
```

A non-zero-variance group resets the streak.

---

# 6.3 Easy vs hard zero-variance groups

Paper deployed thresholds:

```yaml
easy_reward: 1.0
hard_reward: 0.1
```

A group is **easy zero-variance** iff:

```python
all(r == 1.0 for r in rewards)
```

A group is **hard zero-variance** iff:

```python
all(r == 0.1 for r in rewards)
```

Conceptually:

```text
easy:
model consistently solves the prompt

hard:
answer is extractable but consistently wrong
```

A zero-variance reward group not matching either definition MUST NOT silently be mapped to easy or hard.

Store it as:

```text
other
```

Under the frozen reward semantics, this includes every all-`0.0` extraction-failure group.

See Section 15 for the paper ambiguity and the reviewed SignalForge resolution.

---

# 6.4 Reward-history score

For a prompt whose current trailing zero-variance type is (\tau):

[
s_{\mathrm{rew}}(x_i)
=====================

\max
\left(
p_{e,\tau}^{,z_i},
\epsilon_p
\right).
]

Paper deployed defaults:

```yaml
p_easy_initial: 0.5
p_hard_initial: 0.5
epsilon_p: 0.01
p_min: 0.05
p_max: 0.95
```

Example:

```text
p_easy = 0.5

z = 0 -> 1.000
z = 1 -> 0.500
z = 2 -> 0.250
z = 3 -> 0.125
z = 4 -> 0.0625
z = 5 -> 0.03125
...
floor -> 0.01
```

This is deliberately probabilistic.

Repeatedly saturated prompts become less likely to receive rollout compute but are never permanently removed.

---

# 6.5 Unseen prompts

An unseen prompt has:

[
z_i=0.
]

Therefore:

[
s_{\mathrm{rew}}=1.
]

Implementation invariant:

> Every unseen prompt receives a free first visit.

Stage 1 must not reject an unseen prompt in the deployed `lambda = 1` configuration.

Unit-test this explicitly.

---

# 6.6 Historical response entropy

The general HIVE formulation also defines historical response entropy.

For rollout response (r):

[
U_i^{(r)}
=========

\frac{1}{L_{i,r}}
\sum_{\ell=1}^{L_{i,r}}
\mathcal H
\left(
p_\theta(
\cdot
\mid
x_i,o_{i,<\ell}^{(r)}
)
\right)
]

and group-level historical response entropy:

[
H_i
===

\frac{1}{G}
\sum_{r=1}^{G} U_i^{(r)}.
]

The normalized score is:

[
s_{\mathrm{ent}}(x_i)
=====================

\frac{
H_i-H_{\min}
}{
H_{\max}-H_{\min}+\epsilon
}.
]

Normalization is performed over the current metadata pool.

### IMPORTANT

The deployed main HIVE configuration uses:

```yaml
lambda: 1.0
```

so:

[
P_{\mathrm{S1}} = s_{\mathrm{rew}}
]

and historical response entropy has zero weight in the main selector.

Therefore:

1. the architecture should support storing / computing historical response entropy;
2. it should be available for diagnostics and ablations;
3. failure to implement response entropy must NOT be replaced with a fake approximation;
4. the first main HIVE integration may keep this field nullable while `lambda=1`;
5. before claiming the response-entropy ablation is reproduced, exact entropy plumbing must be implemented.

Do not delay the main deployed HIVE path merely because historical response entropy is not yet available.

---

# 6.7 Stage-1 combined probability

General formula:

[
P_{\mathrm{S1}}(x_i)
====================

\lambda s_{\mathrm{rew}}(x_i)
+
(1-\lambda)s_{\mathrm{ent}}(x_i).
]

Paper deployed main configuration:

```yaml
lambda: 1.0
```

Therefore default behavior:

```python
p_select = reward_history_score
```

The entropy term should remain configurable for ablation purposes.

---

# 6.8 Bernoulli selection

For each raw prompt:

```python
accept = bernoulli(P_S1)
```

Use the same global experiment seed family as the trainer.

Paper default seed:

```yaml
seed: 42
```

Resume behavior MUST be deterministic.

Saving only:

```text
p_easy
p_hard
```

is insufficient.

Checkpoint/resume must preserve enough RNG state or deterministic indexing information so Stage-1 selection does not silently change after resume.

---

# 7. Adaptive Easy/Hard Exploration

Paper reference: Eq. (4), Appendix B.3.

The selector dynamically adjusts separate exploration probabilities:

```text
p_easy
p_hard
```

Initial values:

```yaml
p_easy: 0.5
p_hard: 0.5
```

Step size:

```yaml
delta_p: 0.01
```

Bounds:

```yaml
p_min: 0.05
p_max: 0.95
```

Total desired zero-variance fraction:

[
\alpha=0.25.
]

Deployed split targets:

[
\alpha_{\mathrm{easy}}
======================

\frac{\alpha}{3}
\approx0.0833
]

[
\alpha_{\mathrm{hard}}
======================

\frac{2\alpha}{3}
\approx0.1667.
]

The hard target is deliberately larger so the selector continues revisiting prompts that are currently difficult but may later become learnable.

Update each probability after observing the current rollout statistics:

[
p_\tau^{t+1}
============

\operatorname{clip}
\left(
p_\tau^t
+
\Delta p
\cdot
\operatorname{sign}
(
\alpha_\tau-\hat\rho_\tau^t
),
p_{\min},
p_{\max}
\right).
]

Interpretation:

```text
observed zero-var ratio too high
→ reduce corresponding exploration probability

observed zero-var ratio below target
→ increase corresponding exploration probability
```

Track and log both values every training step.

---

# 8. Stage 2 — Current-Policy Prompt Entropy

Paper reference: Section 3.3, Eq. (7)–(9), Appendix B.3.

Stage 2 exists because Stage-1 history becomes stale as the actor learns.

It must use the **current actor policy**.

---

# 8.1 Prompt entropy definition

For tokenized prompt

[
x=(x_1,\ldots,x_{L_x}),
]

compute:

[
V_t(x)
======

\frac{1}{L_x-1}
\sum_{\ell=2}^{L_x}
\mathcal H
\left(
p_{\theta_t}
(
\cdot\mid x_{<\ell}
)
\right).
]

Categorical entropy:

[
\mathcal H(p)
=============

-\sum_{v\in\mathcal V}p_v\log p_v.
]

Operational interpretation:

```text
teacher-force the prompt through the CURRENT actor
→ obtain next-token distribution at each valid prompt position
→ calculate full categorical entropy
→ exclude padding
→ average across valid predictive positions
```

No response generation is required.

---

# 8.2 Token masking

For a prompt containing (L) non-padding tokens:

* there are (L-1) valid next-token prediction positions;
* the first token has no preceding prompt context and is excluded;
* padding tokens must never contribute.

The implementation must be tested with left and/or right padding according to the tokenizer configuration actually used.

Do not accidentally average entropy across padded positions.

---

# 8.3 Full-distribution entropy

Prompt entropy requires the categorical distribution over the vocabulary.

It is NOT equivalent to:

```text
negative log probability of the observed prompt token
```

and NOT equivalent to:

```text
top-1 log probability
```

or:

```text
sampling logprob returned by vLLM
```

unless the implementation can mathematically reconstruct the complete entropy.

Correct conceptual implementation:

```python
with torch.no_grad():
    logits = actor(input_ids, attention_mask).logits

    log_p = log_softmax(logits.float(), dim=-1)
    p = exp(log_p)

    token_entropy = -(p * log_p).sum(dim=-1)
```

Then apply the valid prompt-position mask and average per prompt.

---

# 8.4 Memory requirements for entropy forward

Do NOT retain a full:

```text
[B, L, vocab]
```

tensor longer than necessary.

Potential engineering strategies:

* entropy micro-batching;
* immediately reduce vocabulary dimension;
* immediately detach and move scalar per-token/per-prompt results;
* avoid gradient graph creation;
* reuse actor forward infrastructure where possible.

Do not implement an approximation solely to reduce memory unless explicitly approved and documented as a deviation.

---

# 9. Stage-2 Entropy Band Gate

The deployed v2 configuration contains an important detail beyond the simple median gate.

Defaults:

```yaml
k_off: 0.25
k_keep: 0.50
```

For each Stage-2 selection pool:

1. calculate current prompt entropy;
2. sort candidates by entropy descending;
3. discard the highest-entropy `k_off` fraction;
4. retain the next `k_keep` fraction;
5. discard the remaining low-entropy tail.

Therefore the deployed selector is conceptually a **middle-high entropy band**, not simple entropy maximization.

With:

```text
k_off  = 25%
k_keep = 50%
```

the retained region corresponds approximately to:

```text
25th–75th percentile when viewed from highest to lowest
```

or equivalently the middle 50% after removing both the extreme high-entropy and low-entropy regions.

The paper motivates upper trimming because extremely high entropy can correspond to degenerate / non-reasoning prompts early in training.

---

## 9.1 Integer count and G-rounding semantics

Let `N` be the number of prompts entering one Stage-2 selection pool. Convert the configured fractions to prompt counts using explicit floor semantics:

```text
upper_trim_count     = floor(N * upper_trim_ratio)
pre_round_keep_count = floor(N * keep_ratio)
post_round_keep_count =
    floor(pre_round_keep_count / G) * G
```

Apply these counts after sorting by entropy descending:

1. the first `upper_trim_count` prompts are upper-trimmed;
2. the next `pre_round_keep_count` prompts form the pre-round retained entropy band;
3. retain the first `post_round_keep_count` prompts from that band;
4. remove the remaining prompts from the tail of that band.

Because the band is entropy-sorted descending, Step 4 removes the lowest-entropy prompts within the pre-round retained band.

Prompts removed only by this `G`-multiple rule MUST be classified separately as:

```text
rounding_dropped
```

They MUST NOT be merged into the low-entropy rejection category. Small pools may therefore produce `post_round_keep_count = 0`; do not force at least `G` prompts to survive at the pure selector layer.

---

## 9.2 Deterministic tie handling

Entropy ties must not create nondeterministic batch composition.

Recommended rule:

```text
sort by:
    (-entropy, stable_prompt_id)
```

This is an engineering determinism requirement.

Document if existing veRL distributed ordering requires another deterministic strategy.

---

# 10. Candidate Accumulation

Paper deployed values for major model settings:

[
B_{\mathrm{cand}}
=================

1.5B_t.
]

Original paper-scale configuration:

```yaml
B_t: 256
B_cand: 384
b_raw: 32
```

These exact batch sizes are NOT automatically appropriate for the single-GPU SignalForge reproduction.

However, the ratio:

```text
B_cand / B_t = 1.5
```

should remain the default scaling rule unless memory/performance experiments force a documented deviation.

Example scaled configuration:

```yaml
B_t: TBD
B_cand: 1.5 * B_t
```

The final values belong in `EXPERIMENT_PROTOCOL.md`, not hard-coded into HIVE.

For the faithful main reproduction, derive the initial candidate target exactly:

```text
B_cand_target = 3 * B_t / 2
```

Require `3 * B_t` to be divisible by `2`; do not round a fractional target or expose an
independent candidate-target override.

Apply Appendix B.3 using per-raw-batch Stage-2 pools:

1. fetch one `b_raw` prompt batch;
2. run Stage 1 and then Stage 2 on that batch;
3. append the complete Stage-2 `kept` partition to `C_t` in deterministic raw-batch arrival order;
4. repeat until `len(C_t) >= B_cand_target`.

`B_cand_target` is a lower bound. If the final kept partition crosses the target, retain that entire partition and send
every retained prompt to rollout. Do not truncate, rerank across raw batches, or create a boundary-dropped category.

The resulting `candidate_actual` may exceed the target because Stage 2 rounds each kept partition to a multiple of `G`.
Log the target, actual count, overshoot, actual ratio to `B_t`, and accumulation-round count. Do not compensate for
this discretization in code.

---

# 11. Rollout Phase

After Stage-2 selection:

```text
selected prompts
→ standard vLLM generation
→ G = 8 responses each
→ reward scoring
```

HIVE should reuse the existing SignalForge rollout and reward infrastructure whenever correctness can be preserved.

Do not write a second independent GRPO rollout engine.

---

# 12. Post-Rollout Zero-Variance Filtering

HIVE still performs post-rollout filtering.

For every generated prompt group:

```python
if reward_variance == 0:
    discard_from_training_batch()
else:
    keep()
```

This may appear redundant after Stage 1/2, but it is required.

HIVE reduces the probability of wasting rollouts; it does not guarantee every selected prompt will be effective.

Every generated group — including rejected groups — MUST still count toward rollout-compute accounting.

This is essential.

Do not count only training responses.

---

# 13. Adaptive Top-Up

The final GRPO update needs exactly (B_t) non-zero-variance prompt groups.

If HIVE initially obtains fewer, it must acquire additional candidates.

Define:

```text
effective_responses = number of responses belonging to retained non-zero-var groups
required_responses  = B_t * G
```

If:

[
|\mathcal R_t| < B_tG,
]

paper top-up rule:

[
B_{\mathrm{cand}}^{\mathrm{adapt}}
==================================

\max
\left(
b_{\min},
\min
\left(
B_{\mathrm{cand}},
\left\lceil
\frac{
\eta(B_tG-|\mathcal R_t|)
}{
(1-\hat\rho_{\mathrm{zv}})G
}
\right\rceil
\right)
\right).
]

Defaults:

```yaml
b_min: 64
eta: 1.25
```

Interpretation:

```text
remaining effective groups required
/
estimated survival probability
*
safety factor
```

Then perform another candidate-selection / rollout cycle.

Repeat until enough effective prompt groups have been collected.

---

# 13.1 Required numerical guards

The paper formula becomes unstable when:

[
\hat\rho_{\mathrm{zv}}\rightarrow1.
]

Implementation MUST guard against:

* division by zero;
* infinite top-up loop;
* dataset exhaustion;
* repeated rejection of all candidates.

Suggested engineering safety:

```text
max_selector_rounds
min_survival_epsilon
explicit failure with diagnostics
```

These are safety guards, not algorithmic heuristics.

If triggered during formal training, the run should be considered unhealthy and investigated rather than silently changing the algorithm.

---

# 14. Final Batch Slicing

Once the effective pool contains at least (B_t) groups:

```text
keep first B_t COMPLETE prompt groups
```

and pass those groups to GRPO.

Do NOT slice flattened responses before grouping.

Correct:

```text
effective_prompt_groups[:B_t]
→ flatten
→ GRPO
```

Incorrect:

```text
all_effective_responses[:B_t * G]
```

if ordering could split or mix prompt groups.

Arrival ordering should remain deterministic.

---

# 15. Paper Ambiguities and Resolutions

This section is mandatory because HIVE v2 contains several inconsistencies between the main methodology, detailed appendix, and algorithm prose.

Codex MUST NOT resolve these independently.

## 15.1 Stage-2 median gate vs entropy band

### Main Section 3.3

Describes:

```text
keep entropy >= median
```

which corresponds to keeping the top 50%.

### Appendix B.3 deployed configuration

Adds:

```text
k_off = 0.25
k_keep = 0.50
```

and explicitly says:

```text
sort descending
drop highest 25%
keep next 50%
```

### SignalForge resolution

Use **Appendix B.3 deployed entropy band** for faithful main reproduction.

Implement a configuration switch allowing:

```yaml
upper_trim_ratio: 0.0
keep_ratio: 0.5
```

to reproduce the simpler Eq. (9) median-gate ablation.

Default main reproduction:

```yaml
upper_trim_ratio: 0.25
keep_ratio: 0.50
```

---

## 15.2 Candidate ratio: 1.5× vs 2×

Appendix B.3 and Algorithm 1's inline comment specify:

[
B_{\mathrm{cand}}=1.5B_t.
]

A later prose explanation in Appendix F describes a `2 × B_t` pool.

### SignalForge resolution

Use:

[
B_{\mathrm{cand}}=1.5B_t
]

because Appendix B.3 explicitly describes the deployed main-run configuration.

Record this decision in `HIVE_DEVIATIONS.md` as a paper-internal ambiguity, not as a deviation from the chosen implementation source.

---

## 15.3 Shared alpha vs separate easy/hard targets

Main Eq. (4) is written using a generic target (\alpha).

Appendix B.3 defines deployed targets:

```text
alpha_total = 0.25
alpha_easy  = alpha / 3
alpha_hard  = 2 alpha / 3
```

### SignalForge resolution

Use the separate deployed targets.

---

## 15.4 Reward-score floor

The simplified Stage-1 equation gives approximately:

[
s_{\mathrm{rew}}=p_e^z.
]

Appendix B.3 adds:

[
s_{\mathrm{rew}}
================

\max(p_e^z,\epsilon_p)
]

with:

```yaml
epsilon_p: 0.01
```

### SignalForge resolution

Use the deployed epsilon floor.

---

## 15.5 Historical response entropy

The general HIVE method describes Stage 1 as combining:

```text
reward history
+
historical response entropy
```

but the deployed main configuration sets:

```yaml
lambda: 1.0
```

so historical entropy contributes zero weight.

### SignalForge resolution

Implement the general interface.

Default:

```yaml
lambda: 1.0
```

Do not require historical response entropy for the first correct main-path implementation.

Implement exact historical entropy before running entropy-weight ablations.

---

## 15.6 Reward 0.0 / extraction failure

The paper explicitly identifies:

```text
1.0 = correct
0.1 = extracted but incorrect
```

and defines easy/hard zero-variance groups through those values.

The handling of a fully unextractable zero-variance group is not sufficiently specified for our implementation purposes.

### SignalForge resolution

SignalForge freezes the shared A/B reward semantics as:

```text
correct                   -> 1.0
extractable but incorrect -> 0.1
extraction failure        -> 0.0
```

Zero-variance groups are classified exactly as:

```text
all 1.0 -> easy
all 0.1 -> hard
any other constant group, including all 0.0 -> other
```

The `other` exploration probability is the fixed configurable `p_default=0.5`; see `HIVE_DEVIATIONS.md`.

---

## 15.7 Stage-2 pooling location

Algorithm-level description suggests:

```text
accumulate Stage-1 candidates
→ Stage 2 over full candidate set
```

while the Appendix B.3 exact batch-sizing description can be read as applying Stage 2 during repeated raw-batch accumulation.

### SignalForge resolution

Use scheme B and follow Appendix B.3 exact batching:

```text
for each b_raw batch:
    Stage 1
    Stage 2
    append the complete Stage-2-kept partition
until len(C_t) >= B_cand_target
```

Do not pool Stage-1 survivors across raw batches before Stage 2.

---

# 16. Reward Adapter Requirements

HIVE selection and zero-variance classification depend directly on reward semantics.

The reward adapter therefore becomes part of the algorithm, not merely evaluation infrastructure.

Required output per response:

```python
RewardResult:
    reward: float
    extracted: bool
    correct: bool
```

Frozen HIVE reward semantics:

```text
correct                   -> 1.0
extractable but incorrect -> 0.1
extraction failure        -> 0.0
```

Do not collapse these three outcomes into binary `0/1`; baseline A and HIVE B must use the same adapter.

---

# 17. Checkpoint / Resume Requirements

A HIVE checkpoint must preserve training state AND selector state.

At minimum:

```text
actor / optimizer state
scheduler state if present
global training step

prompt history
p_easy
p_hard
selector RNG state
dataset / dataloader resume state if required
HIVE configuration
```

Resume test:

```text
Run A:
steps 0–10 continuously

Run B:
steps 0–5
save
resume
steps 6–10
```

Given deterministic generation assumptions where possible, selector-level behavior should match:

```text
Stage-1 probability
Stage-1 Bernoulli decisions
p_easy
p_hard
history streaks
```

Any unavoidable generation nondeterminism should be separated from selector nondeterminism.

---

# 18. Required Metrics

HIVE cannot be evaluated only by validation accuracy.

## 18.1 Dataset / candidate metrics

Log per step:

```text
hive/raw_prompts_seen
hive/unseen_prompts_seen

hive/stage1_accepted
hive/stage1_rejected
hive/stage1_accept_ratio

hive/stage2_input
hive/stage2_kept
hive/stage2_upper_trimmed
hive/stage2_low_entropy_rejected
hive/stage2_keep_ratio
```

---

## 18.2 Entropy metrics

```text
hive/prompt_entropy_mean
hive/prompt_entropy_std
hive/prompt_entropy_min
hive/prompt_entropy_max

hive/prompt_entropy_q25
hive/prompt_entropy_q50
hive/prompt_entropy_q75

hive/selected_entropy_mean
hive/rejected_entropy_mean
```

When historical response entropy is implemented:

```text
hive/response_entropy_history_mean
hive/prompt_response_entropy_correlation
```

---

## 18.3 Zero-variance metrics

```text
hive/easy_zero_var_groups
hive/hard_zero_var_groups
hive/other_zero_var_groups
hive/total_zero_var_groups

hive/easy_zero_var_ratio
hive/hard_zero_var_ratio
hive/total_zero_var_ratio
```

Also log:

```text
hive/p_easy
hive/p_hard
```

---

## 18.4 Training-efficiency metrics

Cumulative:

```text
compute/generated_prompt_groups
compute/generated_responses
compute/generated_response_tokens

compute/effective_prompt_groups
compute/effective_responses
compute/effective_response_tokens
```

Derived:

```text
effective_prompt_groups / generated_prompt_groups

effective_responses / generated_responses

effective_training_tokens / generated_tokens

effective_prompt_groups per 1M generated tokens
```

This accounting must include discarded rollout groups.

---

## 18.5 Timing metrics

Separate:

```text
time/stage1
time/stage2_entropy
time/rollout
time/reward
time/topup
time/grpo_update
time/validation
time/checkpoint
time/iteration_total
```

Do not hide selector cost inside rollout time.

---

## 18.6 Top-up metrics

```text
hive/topup_rounds
hive/topup_candidate_target
hive/estimated_zero_var_ratio
hive/effective_groups_before_topup
hive/effective_groups_after_topup
```

---

# 19. Primary Evaluation Curves

Formal comparison should eventually include:

```text
validation accuracy vs training step
validation accuracy vs generated responses
validation accuracy vs generated response tokens
validation accuracy vs rollout wall-clock time
validation accuracy vs total wall-clock time
```

Also:

```text
effective prompt ratio vs step
zero-variance ratio vs step
p_easy / p_hard vs step
prompt entropy quantiles vs step
```

For final reporting, save both:

```text
best checkpoint
final checkpoint
```

Do not report only the best checkpoint.

---

# 20. Dataset Requirements

Dataset choice is currently intentionally UNFROZEN.

Candidates:

```text
A. MATH
B. DAPO + MATH
```

Open-R1 30K is not required for the first reproduction.

Before formal training, run a calibration pass with the final base model:

```text
Qwen2.5-3B
temperature = formal temperature
G = 8
```

For a representative prompt sample, measure:

```text
0/8
1/8
2/8
3/8
4/8
5/8
6/8
7/8
8/8
```

and:

```text
zero-variance ratio
easy zero-var ratio
hard zero-var ratio
extraction-failure ratio
response length distribution
generated-token distribution
```

Do not use only aggregate `mixed-group ratio` to judge dataset difficulty.

Dataset must be frozen in `EXPERIMENT_PROTOCOL.md` before formal A/B experiments.

---

# 21. Model / Training Configuration

Project-fixed model:

```yaml
model: Qwen2.5-3B
```

Paper-method defaults to preserve where practical:

```yaml
rollout_temperature: 1.0
group_size: 8

optimizer: AdamW
learning_rate: 1e-6
beta1: 0.9
beta2: 0.999
weight_decay: 0.01

kl_penalty: disabled
```

Paper-scale values such as:

```text
B_t = 256
B_cand = 384
1000 steps
8 × A100-80GB
```

must NOT be blindly copied to the single-GPU setup.

Single-GPU scaling decisions belong in `EXPERIMENT_PROTOCOL.md`.

Preserve algorithmic ratios and semantics where possible.

---

# 22. Prompt Template

The paper uses a fixed math-reasoning prompt and requires the final answer in boxed form.

Do not allow each dataset loader to silently produce different instructions.

Store the final prompt template centrally in configuration.

The paper's visible template begins with the equivalent of:

> “Please solve the following math problem: {{Question Description}}”

and requires the final answer to be returned in `\boxed{}` form.

Before formal experiments, copy the exact verified paper template into the protocol/config and freeze it.

---

# 23. Recommended Software Components

Names are conceptual; Codex should adapt them to the existing repository instead of creating unnecessary parallel abstractions.

Suggested responsibilities:

```text
HiveHistoryStore
    persistent per-prompt metadata

HiveStage1Selector
    reward-history score
    optional entropy score
    Bernoulli selection

PromptEntropyEvaluator
    current-policy prompt entropy

HiveStage2Selector
    entropy sorting
    upper trim
    keep band

HiveExplorationController
    p_easy / p_hard adaptation

HiveBatchAccumulator
    candidate accumulation
    rollout filtering
    top-up
    exact final batch construction

HiveMetrics
    selector / compute accounting
```

Prefer small testable components over one giant modification to the trainer loop.

---

# 24. Required Unit Tests

## 24.1 History streak

Test:

```text
NZ -> z=0
Z -> z=1
Z,Z -> z=2
Z,Z,NZ -> z=0
Z,Z,NZ,Z -> z=1
```

---

## 24.2 Unseen prompt

Verify:

```text
history missing
→ z=0
→ reward score=1
→ with lambda=1, Stage-1 acceptance probability=1
```

---

## 24.3 Reward-score decay

Test:

```text
p=0.5
z=1 -> 0.5
z=2 -> 0.25
...
```

and floor:

```text
score >= 0.01
```

---

## 24.4 Exploration controller

Verify:

```text
rho_easy > alpha_easy
→ p_easy decreases

rho_easy < alpha_easy
→ p_easy increases
```

Same for hard.

Verify clipping:

```text
0.05 <= p <= 0.95
```

---

## 24.5 Easy/hard classification

```text
[1.0] * 8
→ easy

[0.1] * 8
→ hard

mixed
→ non-zero-var

[0.0] * 8
→ other
```

---

## 24.6 Prompt entropy padding

Same prompt:

```text
alone
vs
inside differently padded batch
```

must yield numerically equivalent entropy.

---

## 24.7 Prompt entropy correctness

For a tiny model / tiny vocabulary test case:

compare implementation against a direct reference computation of:

[
-\sum p\log p.
]

---

## 24.8 Entropy uses current actor

Perturb actor weights in a controlled unit/integration test.

Verify prompt entropy changes.

This protects against accidental use of a stale model.

---

## 24.9 Stage-2 entropy band

Construct deterministic entropy values:

```text
[10, 9, 8, 7, 6, 5, 4, 3]
```

With:

```text
k_off = 0.25
k_keep = 0.50
```

verify:

```text
10,9 discarded as upper extreme
8,7,6,5 form the pre-round retained entropy band
4,3 discarded as low entropy
```

Then verify the approved integer semantics. With the default `G=8`:

```text
upper_trim_count = floor(8 * 0.25) = 2
pre_round_keep_count = floor(8 * 0.50) = 4
post_round_keep_count = floor(4 / 8) * 8 = 0

8,7,6,5 -> rounding_dropped
final kept prompts -> empty
```

Also test a pool where `pre_round_keep_count` is larger than `G` but not a multiple of `G`; verify that rounding removes the lowest-entropy tail of the pre-round retained band and reports the loss separately from low-entropy rejection.

---

## 24.10 Group integrity

Top-up and slicing must never break an eight-response group.

---

## 24.11 Budget accounting

Generate synthetic groups where some are discarded.

Verify:

```text
generated_response_count
>
training_response_count
```

and both values are logged correctly.

---

## 24.12 Resume selector state

Save and restore:

```text
history
p_easy
p_hard
RNG
```

Verify selector decisions remain reproducible.

---

# 25. Implementation Order

Codex MUST NOT implement everything in one patch.

Recommended sequence:

## Phase 0 — Architecture only

Read:

```text
AGENTS.md
EXPERIMENT_PROTOCOL.md
HIVE_IMPLEMENTATION_SPEC.md
existing SignalForge trainer
existing Dynamic Sampling / replenish code
```

Produce an implementation map.

Do not modify code.

---

## Phase 1 — History infrastructure

Implement:

```text
prompt_id
PromptHistory
reward trace
zero-var type
trailing zero-var streak
checkpoint persistence
```

Unit tests first.

No Stage 2 yet.

---

## Phase 2 — Stage 1

Implement:

```text
reward-history score
epsilon floor
Bernoulli selection
p_easy / p_hard adaptation
```

Default:

```text
lambda = 1
```

Test independently.

---

## Phase 3 — Prompt entropy evaluator

Implement independently from selector logic.

Verify:

```text
padding
numerical correctness
current-policy weights
memory usage
micro-batching
```

Do not integrate into full HIVE until this module passes tests.

---

## Phase 4 — Stage 2

Implement:

```text
entropy scoring
upper trim
keep band
stable ordering
metrics
```

---

## Phase 5 — Rollout integration

Connect:

```text
Stage 1
→ Stage 2
→ existing vLLM rollout
→ existing reward adapter
```

---

## Phase 6 — Zero-var filtering + top-up

Reuse correct SignalForge v1 replenish logic where possible.

Implement HIVE-specific adaptive candidate estimate.

Verify fixed effective batch size.

---

## Phase 7 — Selector checkpoint/resume

Run an end-to-end resume smoke test.

---

## Phase 8 — Instrumentation

Add all required HIVE and compute-efficiency metrics.

No formal experiment should start without these.

---

# 26. Smoke-Test Protocol

First HIVE smoke:

```text
tiny dataset subset
2–5 optimizer steps
G=8
formal prompt template
formal reward adapter
```

Success means:

* no crash;
* Stage-1 state changes correctly;
* Stage-2 entropy is finite;
* candidates are actually filtered;
* rollouts occur only after Stage 2;
* zero-var groups are removed;
* top-up can fill a batch;
* GRPO receives complete groups;
* history updates after rollout;
* checkpoint save works;
* resume works.

Accuracy is irrelevant.

---

# 27. Pilot Protocol

Before formal training, run approximately 50–100 steps.

Inspect mechanism behavior rather than benchmark gains.

Questions:

```text
Is p_easy decreasing as easy prompts saturate?

Does p_hard remain higher than p_easy?

How many prompts does Stage 1 reject?

What entropy region does Stage 2 retain?

Are extreme-high entropy prompts actually different?

Does Stage 2 reduce downstream zero-var rate?

How many rollout groups are still discarded?

How many top-up rounds are needed?

How expensive is prompt entropy relative to rollout?

Does every update receive exactly B_t effective groups?

Does HIVE reduce generated responses/tokens per update?
```

If these mechanisms do not behave sensibly, do not launch the formal run.

---

# 28. Formal Experiment Skeleton

Exact protocol remains to be frozen later.

Minimum comparison:

```text
A — vanilla GRPO
B — HIVE
```

Preferred if budget permits:

```text
A — vanilla GRPO
B — Dynamic Sampling
C — HIVE
```

All comparisons must control:

```text
model
dataset
reward
prompt format
rollout temperature
group size
optimizer
LR
validation
maximum response length
training-update definition
```

Report compute separately rather than pretending equal optimizer steps imply equal rollout cost.

---

# 29. Deviation Logging

Create:

```text
docs/hive/HIVE_DEVIATIONS.md
```

Every deliberate difference from the paper must contain:

```text
Paper behavior:
SignalForge behavior:
Reason:
Expected consequence:
Affects algorithm claim? yes/no
```

Known project-level deviation already:

```text
Paper tested several model families but not plain Qwen2.5-3B.
SignalForge uses Qwen2.5-3B.
```

This is acceptable, but must be documented.

Expected future deviations:

```text
single GPU instead of 8×A100
smaller logical batch
possibly shorter training horizon
possibly different context length
dataset choice
evaluation-suite scale
```

Do not hide these.

---

# 30. Codex Rules

For any HIVE implementation task:

1. Read this file first.
2. Read `EXPERIMENT_PROTOCOL.md`.
3. Inspect the existing code path before proposing changes.
4. Prefer reusing existing SignalForge / veRL functionality.
5. Do not duplicate the GRPO trainer.
6. Do not introduce SARA or VIGOR behavior.
7. Do not silently change an equation or hyperparameter.
8. Do not hard-code machine-specific paths.
9. Every algorithmic modification must identify the corresponding section of this specification.
10. Any paper ambiguity must be surfaced before implementation.
11. Any implementation deviation must be added to `HIVE_DEVIATIONS.md`.
12. Tests are required before integration.
13. Formal experiments must not begin until compute accounting is verified.

---

# 31. Definition of Done

The HIVE implementation is considered complete only when all of the following are true:

* stable prompt history exists;
* unseen prompts receive the intended first visit;
* trailing zero-variance history works;
* easy/hard exploration probabilities adapt correctly;
* Stage-1 Bernoulli selection is deterministic under checkpoint/resume;
* current-policy prompt entropy is computed correctly;
* padding does not affect entropy;
* Stage-2 entropy-band filtering works;
* rollout happens after selection;
* zero-variance groups are rejected after rollout;
* adaptive top-up fills the fixed effective batch;
* groups remain intact;
* HIVE state survives checkpoint/resume;
* generated rollout cost is counted before filtering;
* selector overhead is timed separately;
* smoke tests pass;
* mechanism pilot passes;
* deviations from the paper are documented;
* formal experiment protocol is frozen before expensive training begins.

Only after these conditions hold should SignalForge v2 begin formal GRPO / DS / HIVE comparison.
