# Signal Forge Experiment Protocol

## 0. Document Status

- **Project:** Signal Forge
- **Primary model:** `Qwen2.5-1.5B-Instruct`
- **Training framework:** `veRL`
- **Task type:** Mathematical RLVR
- **Current implementation stage:** Experiment A — GRPO Baseline
- **Protocol version:** `v1.0`
- **Protocol status:** Draft for implementation
- **Last updated:** 2026-07-28

> This document is the source of truth for all Signal Forge experiments.
> Any change that affects model, data, reward, rollout, optimization, evaluation,
> or compute budget must be recorded here before the corresponding run starts.

---

## 1. Research Question

Under a fixed rollout-token budget, which training prompts are most valuable for
reinforcement learning with verifiable rewards on a 1.5B language model?

More specifically, this project studies whether:

1. removing prompts whose rollout groups are entirely correct or entirely wrong;
2. relaxing clipping for positive-advantage updates;
3. penalizing overlong responses;
4. adaptively sampling prompts according to pass rate, raw reward variance, and
   learning progress;

can improve the sample efficiency, training stability, or final mathematical
reasoning performance of `Qwen2.5-1.5B-Instruct`.

The project does **not** aim to claim a new general-purpose RL algorithm.
Its goal is to build a controlled, reproducible empirical study of training-signal
quality under a limited compute budget.

---

## 2. Core Experimental Principle

The main comparison metric is:

> **Validation performance under the same cumulative rollout-token budget.**

Optimizer steps alone are not a fair compute measure because Dynamic Sampling
and Adaptive Curriculum Sampling may generate and discard additional rollout
groups.

Every experiment must therefore report at least:

- optimizer steps;
- prompt groups generated;
- responses generated;
- accepted prompt groups;
- cumulative rollout tokens;
- rollout tokens per optimizer step;
- GPU hours;
- wall-clock time.

---

## 3. Fixed Project Configuration

The following settings are fixed across Experiments A–E unless this document is
explicitly amended.

### 3.1 Model

```yaml
model:
  name_or_path: Qwen/Qwen2.5-1.5B-Instruct
  model_family: Qwen2.5
  parameter_scale: 1.5B
  initial_checkpoint: pretrained_instruct_checkpoint
```

All formal comparisons must start from the same initial checkpoint.

### 3.2 Prompt Format

```yaml
prompt:
  style: zero-shot
  final_answer_requirement: boxed
  expected_pattern: "\\boxed{...}"
```

The exact prompt template must be version-controlled and identical across:

- RewardScope offline analysis;
- Signal Forge training;
- validation evaluation;
- OOD evaluation.

### 3.3 Rollout Configuration

```yaml
rollout:
  group_size: 8
  max_response_length: 768
  temperature: TODO
  top_p: TODO
  top_k: TODO
  do_sample: true
```

Generation parameters must remain fixed across A–E unless the generation policy
itself becomes an explicit experimental variable.

### 3.4 Dataset Mixture

```yaml
dataset:
  training_mixture:
    gsm8k: 0.60
    math_level_3: 0.40
  math_subset: level_3_only
  dataset_version: signal_forge_v1
  split_seed: TODO
```

The processed dataset must preserve:

- source dataset;
- original sample ID;
- problem text;
- reference answer;
- normalized verifier answer;
- difficulty label;
- split assignment;
- dataset version.

### 3.5 Reward and Verifier

```yaml
reward:
  raw_reward_type: binary_correctness
  verifier: boxed_only_math_verify
  correct_value: 1.0
  incorrect_value: 0.0
```

For every response, preserve the following fields even when only the binary
correctness reward is used for optimization:

```text
raw_correctness_reward
extraction_ok
format_ok
predicted_answer
reference_answer
failure_reason
```

The training verifier must be behaviorally identical to the verifier used in
RewardScope and evaluation.

---

## 4. Experiment Matrix

## 4.1 Experiment A — GRPO Baseline

### Purpose

Establish the common baseline for all later experiments.

### Configuration

Experiment A uses:

- random sampling from the frozen training set;
- standard GRPO;
- binary correctness reward;
- no filtering of all-correct or all-wrong rollout groups;
- no asymmetric clipping;
- no length penalty;
- no curriculum sampling.

```yaml
experiment:
  id: A
  name: grpo_baseline

features:
  dynamic_sampling: false
  clip_higher: false
  overlong_reward_shaping: false
  adaptive_curriculum_sampling: false
```

### Hypothesis

Standard GRPO will improve in-domain mathematical accuracy, but a meaningful
fraction of rollout compute may be spent on prompt groups that contain no
within-group correctness variation.

### Required Outputs

Experiment A must establish:

- baseline validation accuracy;
- baseline pass@1;
- all-correct, all-wrong, and mixed-group ratios;
- rollout-token consumption;
- response-length distribution;
- training stability;
- reproducible checkpoint and configuration artifacts.

Experiment A is considered valid only if the full data, reward, rollout,
optimization, logging, evaluation, checkpoint, and resume paths work correctly.

---

## 4.2 Experiment B — GRPO + Dynamic Sampling

### Change From A

For each sampled prompt:

1. generate `n=8` responses;
2. compute raw binary correctness rewards;
3. reject groups that are entirely correct or entirely wrong;
4. retain groups containing both correct and incorrect responses;
5. continue sampling until the target number of accepted groups is reached.

### Important Rule

Filtering must use `raw_correctness_reward`, not length-shaped or otherwise
modified reward.

### Hypothesis

Filtering zero-variance correctness groups increases the proportion of rollout
tokens that produce useful relative-advantage learning signals.

### Additional Required Metrics

- all-correct group ratio;
- all-wrong group ratio;
- mixed-group ratio;
- generated groups per accepted group;
- discarded rollout tokens;
- accepted groups per optimizer step;
- additional rollout cost introduced by rejection sampling.

---

## 4.3 Experiment C — Dynamic Sampling + Clip-Higher

### Change From B

Use asymmetric clipping inspired by DAPO:

- retain stricter clipping for negative-advantage updates;
- use a wider upper clipping bound for positive-advantage updates.

```yaml
clip_higher:
  enabled: true
  clip_low: TODO
  clip_high_positive: TODO
```

### Hypothesis

A wider positive clipping range may allow low-probability but high-value tokens
to receive stronger positive updates, improving exploration and reducing early
policy collapse.

### Additional Required Metrics

- positive-advantage clip fraction;
- negative-advantage clip fraction;
- total clip fraction;
- policy entropy;
- KL divergence;
- response diversity;
- gradient norm;
- instability or divergence events.

---

## 4.4 Experiment D — Clip-Higher + Overlong Reward Shaping

### Change From C

Add a response-length penalty that:

- does not penalize normal-length responses;
- increases gradually near the configured response-length limit;
- penalizes truncated or severely overlong responses more strongly.

The following reward components must be logged separately:

```text
raw_correctness_reward
format_reward
length_penalty
final_shaped_reward
```

### Hypothesis

Length shaping reduces truncation and unproductive verbosity while preserving
correctness.

### Additional Required Metrics

- mean response length;
- P50, P90, and P95 response length;
- overlong-response ratio;
- truncated-response ratio;
- correct-response length distribution;
- incorrect-response length distribution;
- accuracy per generated token;
- pass@1;
- evidence of under-reasoning caused by excessive shortening.

### Length-Penalty Parameters

```yaml
overlong_reward:
  enabled: true
  soft_threshold: TODO
  hard_threshold: TODO
  maximum_penalty: TODO
```

These values must be finalized before Experiment D begins.

---

## 4.5 Experiment E — Adaptive Curriculum Sampling

### Change From D

Prioritize prompts that are currently likely to produce useful learning signal.

The first implementation should use:

- medium historical pass rate;
- high raw correctness variance;
- learning-progress signal;
- uniform exploration/replay.

### Candidate Statistics

For each prompt or difficulty bucket, maintain:

```text
historical_pass_rate
raw_reward_variance
pass_rate_ema
recent_learning_progress
times_sampled
last_sampled_step
```

### Initial Sampling Principle

The selector should favor prompts that are neither consistently solved nor
consistently failed, while preserving a fixed amount of uniform sampling.

```yaml
curriculum:
  enabled: true
  pass_rate_weight: TODO
  reward_variance_weight: TODO
  learning_progress_weight: TODO
  uniform_sampling_ratio: TODO  # recommended design range: 0.10–0.20
```

### Hypothesis

Under a fixed rollout-token budget, selecting prompts with higher estimated
learning value improves early convergence or final validation performance.

### Additional Required Metrics

- sampling share by difficulty bucket;
- accuracy by difficulty bucket;
- selector-score distribution;
- prompt coverage;
- sample concentration;
- uniform replay share;
- sampling distribution over training time;
- cumulative rollout-token efficiency.

---

## 5. Fair-Comparison Constraints

Experiments A–E must use the same:

- initial model checkpoint;
- tokenizer;
- training set;
- validation set;
- OOD set;
- prompt template;
- verifier;
- raw correctness definition;
- group size;
- maximum response length;
- generation temperature;
- top-p and top-k settings;
- optimizer family;
- learning-rate schedule;
- checkpoint selection rule;
- evaluation prompt;
- evaluation decoding configuration.

Any unavoidable deviation must be recorded in both:

1. this protocol;
2. the experiment run summary.

---

## 6. Data Protocol

## 6.1 Dataset Artifacts

Recommended structure:

```text
data/
├── raw/
│   ├── gsm8k/
│   └── math/
├── processed/
│   └── signal_forge_v1/
│       ├── train.parquet
│       ├── validation_id.parquet
│       ├── validation_ood.parquet
│       ├── manifest.json
│       └── statistics.json
└── scripts/
    └── build_signal_forge_dataset.py
```

### 6.2 Dataset Manifest

`manifest.json` must include:

```json
{
  "dataset_version": "signal_forge_v1",
  "build_script_commit": "TODO",
  "split_seed": "TODO",
  "gsm8k_ratio": 0.60,
  "math_level_3_ratio": 0.40,
  "train_size": "TODO",
  "validation_id_size": "TODO",
  "validation_ood_size": "TODO",
  "source_dataset_versions": "TODO",
  "file_hashes": "TODO"
}
```

### 6.3 Split Rules

- Freeze train, in-domain validation, and OOD evaluation splits before formal
  Experiment A begins.
- Preserve original sample IDs.
- Check for duplicate problems and answer leakage across splits.
- Do not change the data mixture between A–E.
- Any data correction creates a new dataset version and invalidates direct
  comparison with runs using the old version.

---

## 7. Verifier Protocol

The verifier must return a structured result rather than only a scalar reward.

Recommended interface:

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardResult:
    raw_correctness_reward: float
    extraction_ok: bool
    format_ok: bool
    predicted_answer: Optional[str]
    reference_answer: str
    failure_reason: Optional[str]
```

### Required Verifier Tests

At minimum, test:

- correct boxed integer;
- incorrect boxed integer;
- missing boxed answer;
- multiple boxed expressions;
- negative number;
- fraction;
- decimal;
- equivalent symbolic expression;
- malformed LaTeX;
- empty response;
- truncated response;
- Math-Verify exception or timeout.

Normal extraction failures are ordinary incorrect answers and must remain
distinguishable in diagnostics. Verifier/library exceptions are correctness bugs
and must fail loudly; they must not be silently converted into ordinary wrong
answers.

---

## 8. Training Configuration

The exact optimization values must be copied from the final veRL configuration
before a formal run is launched.

```yaml
optimization:
  algorithm: GRPO
  optimizer: TODO
  learning_rate: TODO
  lr_scheduler: TODO
  warmup_ratio_or_steps: TODO
  weight_decay: TODO
  max_grad_norm: TODO

batching:
  train_batch_size_prompts: TODO
  actor_micro_batch_size: TODO
  rollout_micro_batch_size: TODO
  gradient_accumulation_steps: TODO

precision:
  dtype: TODO
  gradient_checkpointing: TODO
  fsdp_or_deepspeed_config: TODO

regularization:
  kl_coefficient: TODO
  entropy_coefficient: TODO
  clip_range: TODO
```

No formal result should be reported while any value used by the actual run
remains undocumented.

---

## 9. Compute-Budget Protocol

### 9.1 Primary Budget

```yaml
compute_budget:
  primary_unit: cumulative_rollout_tokens
  target_rollout_tokens: TODO
```

### 9.2 Secondary Resource Measures

Record:

- optimizer steps;
- prompt groups generated;
- responses generated;
- accepted prompt groups;
- discarded prompt groups;
- rollout tokens;
- peak GPU memory;
- rollout time;
- optimizer-update time;
- step time;
- total GPU hours;
- total wall-clock time.

### 9.3 Budget Matching

Primary A–E comparisons should be made at matched cumulative rollout-token
checkpoints.

Where possible, evaluate all runs at common token milestones such as:

```text
25% of target budget
50% of target budget
75% of target budget
100% of target budget
```

The exact milestones must be decided before formal runs begin.

---

## 10. Evaluation Protocol

## 10.1 Evaluation Sets

At minimum:

1. **In-domain validation**
   - reflects the same task family as training;
   - contains no training examples.

2. **OOD evaluation**
   - different dataset, problem type, or difficulty distribution;
   - used to test whether gains generalize beyond the training mixture.

3. **Difficulty-bucket evaluation**
   - report easy, medium, and hard subsets separately when available.

The exact OOD dataset is currently:

```text
TODO: finalize OOD dataset and version
```

## 10.2 Fixed Evaluation Configuration

```yaml
evaluation:
  temperature: TODO
  top_p: TODO
  top_k: TODO
  max_response_length: 768
  samples_per_prompt: TODO
  checkpoint_interval_rollout_tokens: TODO
```

All checkpoints must use the same evaluation configuration.

## 10.3 Primary Metrics

- validation accuracy;
- pass@1;
- final-answer format correctness;
- OOD accuracy;
- accuracy by difficulty bucket.

Optional, if budget allows:

- pass@4;
- pass@8.

## 10.4 Checkpoint Selection

The checkpoint-selection rule must be fixed before reviewing final results.

```yaml
checkpoint_selection:
  primary_metric: TODO
  tie_breaker: TODO
  selection_set: validation_id
```

The test or OOD set must not be used for checkpoint selection.

---

## 11. Unified Logging Schema

## 11.1 Model Performance

- validation accuracy;
- pass@1;
- optional pass@4/pass@8;
- final-answer format accuracy;
- OOD accuracy;
- accuracy by difficulty bucket.

## 11.2 Training State

- policy loss;
- raw reward mean/std;
- shaped reward mean/std;
- advantage mean/std;
- policy entropy;
- KL divergence;
- gradient norm;
- learning rate;
- positive clip fraction;
- negative clip fraction;
- total clip fraction.

## 11.3 Sampling Efficiency

- all-correct group ratio;
- all-wrong group ratio;
- mixed-group ratio;
- accepted groups per generated group;
- generated groups per optimizer step;
- responses generated;
- rollout tokens per optimizer step;
- cumulative rollout tokens;
- raw reward variance;
- samples per difficulty bucket.

## 11.4 Length Metrics

- mean response length;
- P50/P90/P95 response length;
- truncated ratio;
- overlong ratio;
- mean length penalty;
- correct-response length distribution;
- incorrect-response length distribution.

## 11.5 Resource Metrics

- peak GPU memory;
- rollout time;
- optimizer-update time;
- total step time;
- cumulative GPU hours;
- cumulative wall-clock time.

---

## 12. Reproducibility Protocol

Every formal run must save:

```text
run_id
experiment_id
git_commit
dataset_version
dataset_hash
model_checkpoint
full_resolved_config
random_seed
environment_lockfile
CUDA_version
PyTorch_version
veRL_version
vLLM_version
GPU_model
number_of_GPUs
start_time
end_time
WandB_run_url_or_local_log_path
```

Recommended run directory:

```text
outputs/
└── <experiment_id>/
    └── <run_id>/
        ├── resolved_config.yaml
        ├── environment.txt
        ├── dataset_manifest.json
        ├── metrics.jsonl
        ├── checkpoints/
        ├── evaluation/
        └── run_summary.md
```

---

## 13. Random Seeds and Statistical Reliability

### Exploration Stage

- Run all experiments with one seed to validate implementation and estimate
  effect size.

### Confirmation Stage

If compute allows, repeat the most important comparisons with 2–3 seeds.

Priority for repeated seeds:

1. Experiment A;
2. Experiment B;
3. Experiment E.

Report:

- mean;
- standard deviation or range;
- full per-seed results.

Do not describe a small one-seed difference as statistically significant.

```yaml
seeds:
  exploratory: TODO
  confirmation: TODO
```

---

## 14. Smoke-Test Protocol

Before a formal 1.5B run, complete both tests below.

## 14.1 Local or Existing-Environment Regression Test

Use the already validated smaller model only to test the newly connected
pipeline.

Required checks:

- dataset loads correctly;
- prompt template is correct;
- verifier returns expected structured outputs;
- reward is binary for Experiment A;
- group size is 8;
- response cap is 768;
- loss is finite;
- gradient norm is finite;
- metrics are logged;
- checkpoint saving works;
- resume works.

## 14.2 Qwen2.5-1.5B 80GB GPU Smoke Test

Run approximately 1–5 optimizer steps with the formal Experiment A
configuration.

Check:

- peak GPU memory;
- actor memory;
- reference-model memory;
- rollout-engine memory;
- optimizer-state memory;
- rollout token counts;
- step time;
- checkpoint size;
- disk usage;
- evaluation invocation;
- resume from checkpoint.

If OOM occurs, first adjust implementation-level memory settings such as:

- train micro-batch size;
- rollout micro-batch size;
- vLLM GPU-memory utilization;
- CPU offload;
- gradient checkpointing;
- sequence parallelism;
- number of prompts processed concurrently.

Do not silently reduce `group_size=8` or `max_response_length=768`, because those
are fixed experimental settings.

---

## 15. Experiment A Acceptance Criteria

Experiment A may be marked complete only when all of the following are true:

### Pipeline

- [ ] Frozen dataset version is used.
- [ ] Prompt template matches RewardScope.
- [ ] Training and evaluation use the same verifier behavior.
- [ ] Binary reward is correctly logged.
- [ ] Group size is exactly 8.
- [ ] Maximum response length is exactly 768.
- [ ] All later-stage features are disabled.

### Reliability

- [ ] No unexplained NaN or Inf occurs.
- [ ] Checkpoint saving succeeds.
- [ ] Resume from checkpoint succeeds.
- [ ] Resolved configuration is saved.
- [ ] Git commit and environment are recorded.

### Metrics

- [ ] Validation accuracy and pass@1 are reported.
- [ ] All-correct/all-wrong/mixed group ratios are reported.
- [ ] Cumulative rollout tokens are reported.
- [ ] Response-length percentiles are reported.
- [ ] GPU memory and wall-clock time are reported.

### Artifacts

- [ ] Final or selected checkpoint is saved.
- [ ] Evaluation outputs are saved.
- [ ] Run summary is written.
- [ ] WandB or equivalent logs are retained.
- [ ] Failures and deviations are documented.

---

## 16. Stop and Failure Criteria

A run should be stopped and marked invalid if:

- data corruption or split leakage is discovered;
- verifier behavior differs between training and evaluation;
- configuration differs from the protocol without documentation;
- cumulative rollout-token logging is incorrect;
- reward is not binary in Experiment A;
- later-stage features are accidentally enabled in Experiment A;
- repeated NaN/Inf prevents stable training;
- checkpoint or resume is broken;
- the run starts from a different initial checkpoint;
- the dataset version changes during the run.

Early stopping for performance reasons must use a rule decided before examining
the final comparison.

```yaml
early_stopping:
  enabled: TODO
  metric: TODO
  patience_or_budget_rule: TODO
```

---

## 17. Run Summary Template

Each formal run should produce a `run_summary.md` containing:

```markdown
# Run Summary

## Identity
- Experiment:
- Run ID:
- Seed:
- Git commit:
- Dataset version:
- Initial checkpoint:

## Configuration
- Group size:
- Max response length:
- Batch configuration:
- Learning rate:
- KL coefficient:
- Clip settings:
- Reward settings:

## Compute
- Optimizer steps:
- Prompt groups generated:
- Responses generated:
- Accepted groups:
- Rollout tokens:
- GPU hours:
- Wall-clock time:
- Peak GPU memory:

## Results
- In-domain validation accuracy:
- Pass@1:
- OOD accuracy:
- Format accuracy:
- All-correct group ratio:
- All-wrong group ratio:
- Mixed-group ratio:
- Mean/P90/P95 response length:

## Observations
- Training stability:
- Notable failure cases:
- Unexpected behavior:
- Deviations from protocol:

## Conclusion
- Did the run satisfy acceptance criteria?
- Is the run valid for comparison?
- Main finding:
```

---

## 18. Current Pre-Run TODO List

Before the formal Experiment A run, finalize:

- [ ] exact dataset split seed;
- [ ] exact dataset sizes;
- [ ] dataset hashes;
- [ ] generation temperature/top-p/top-k;
- [ ] optimizer and learning rate;
- [ ] batch and micro-batch sizes;
- [ ] KL and clipping parameters;
- [ ] precision and distributed-training settings;
- [ ] formal rollout-token budget;
- [ ] evaluation interval;
- [ ] OOD dataset;
- [ ] checkpoint-selection rule;
- [ ] exploratory seed;
- [ ] environment versions;
- [ ] Experiment A configuration file;
- [ ] verifier unit tests;
- [ ] 0.5B regression test;
- [ ] 1.5B smoke test on the 80GB GPU.

---

## 19. Protocol Change Log

| Version | Date | Change |
|---|---|---|
| v1.0 | 2026-07-28 | Initial Signal Forge protocol; standardized all experiments on Qwen2.5-1.5B-Instruct. |


---

## 20. A0 Online Observability Addendum

A0/A training uses RewardScope only as the frozen Math-Verify verifier package.
The veRL reward adapter remains a thin boundary and does not duplicate parser,
grader, report, plot, or sampler logic.

For the current AutoDL/vLLM path, Math-Verify parsing timeout is disabled inside
the verifier because `signal.alarm()` is only valid in the Python main thread,
while reward calls may run outside that context. This is an engineering
compatibility fix: the reward rule remains boxed-only binary Math-Verify
correctness. Normal extraction failures are logged as incorrect responses;
library/verifier exceptions should fail loudly.

Formal A runs must record reward compute timing. Current veRL logs already expose
`timing_s/agent_loop/compute_score/min`, `timing_s/agent_loop/compute_score/max`,
`timing_s/agent_loop/compute_score/mean`, and `timing_s/reward`. If rare LaTeX
cases make reward computation too slow, add a process-level timeout outside the
adapter; do not rely on `signal.alarm()` and do not silently treat timeout as a
normal wrong answer without a diagnostic field.

Stable custom metric prefixes introduced for A0/A observability:

- `reward/`: raw correctness, extraction, format and final-score summaries.
- `group/`: all-correct/all-wrong/mixed group diagnostics from raw correctness.
- `budget/`: generated candidate rollout tokens and cumulative compute budget.
- `length/`: lightweight response length percentiles and correctness-conditioned length.
- `val/`: deterministic validation aliases and best-checkpoint metadata.

Rollout budget semantics: prompt tokens are counted once per generated response;
response tokens exclude padding and are based on the response mask. Validation
cost is reported separately by validation metrics and is not added to the
training rollout budget counters.
