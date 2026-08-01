# AGENTS.md — Experiment A / GRPO Baseline

## 1. Project status

This repository is an LLM RLVR/post-training research project based on veRL.

The immediate goal is **Experiment A: a clean standard GRPO baseline** using
Qwen2.5-1.5B-Instruct on a fixed mixture of GSM8K and MATH Level 3.

The broader controlled ablation plan is:

- **A — GRPO baseline:** random prompt sampling and standard GRPO.
- **B — A + Dynamic Sampling:** reject all-correct and all-wrong response groups.
- **C — B + Clip-Higher:** asymmetric clipping inspired by DAPO.
- **D — C + Overlong Reward Shaping:** penalize overly long responses before the hard limit.
- **E — D + Adaptive Curriculum Sampling:** change prompt sampling probabilities using
  prompt success rate, reward variance and optionally learning progress, while retaining
  uniform replay.

Only **A0/A** is in scope now. Do not implement B–E until A is stable and frozen. 
Experiment A is the immutable GRPO baseline.
Do not add dynamic sampling, asymmetric clipping,
length shaping, or curriculum logic to Experiment A.

## 2. Current task

Build and run **A0**, a 2–3 optimizer-step smoke test of the Experiment A pipeline.

A0 is successful when the full chain works:

`dataset -> chat prompt -> n=8 rollout -> boxed-only Math-Verify reward -> GRPO advantage -> actor update -> validation -> checkpoint save/reload -> logs`

A0 is not an accuracy experiment. Do not tune based on the 2–3-step learning curve.

## 3. Existing work that must be reused

A previous veRL setup has already:

- run Qwen2.5-0.5B GRPO successfully on a 4090 32 GB GPU;
- reached the Qwen2.5-1.5B GRPO pipeline after several fixes, but eventually OOMed;
- therefore established that most of the training chain already works.

Before editing:

1. Inspect the repository and locate the previous working launch script, Hydra/YAML config,
   data preparation code, reward code and logs.
2. Reuse that path. Do not create a second trainer or a parallel configuration system.
3. Identify exactly where the 1.5B run OOMed: rollout, reference log-probability,
   actor forward/backward, optimizer step, validation or checkpointing.
4. Record the pinned veRL commit/version and the actual configuration schema used by this
   repository. Do not blindly copy keys from a different veRL release.

The intended final hardware is a **single A800 80 GB**. The earlier 4090 32 GB OOM is
historical evidence, not proof that the A800 configuration will fail.

## 4. Frozen RewardScope findings

RewardScope is complete and functionally frozen. Do not extend or refactor it while
working on Experiment A unless a reproducible correctness bug blocks training.

RewardScope established the following:

- The former custom numeric/`####` extractor severely underestimated model ability.
- On the same 128 old generations:
  - old verifier accuracy: 27.34%;
  - Math-Verify evaluation accuracy: 62.50%;
  - old extraction failure: 67.19%;
  - Math-Verify extraction failure: 2.34%.
- A zero-shot boxed greedy sanity check achieved:
  - accuracy: 67.97%;
  - extraction failure: 2.34%;
  - format error: 6.25%;
  - hit-max rate at 512 tokens: 6.25%.
- A GSM8K zero-shot boxed `n=8` microscope achieved:
  - all-wrong: 3.125%;
  - mixed: 70.3125%;
  - all-correct: 26.5625%;
  - pass@1: 72.46%;
  - effective token ratio: 72.72%.
- MATH Level 1–2 was not a useful hard-data component:
  - all-wrong: 10.16%;
  - mixed: 46.88%;
  - all-correct: 42.97%.
- MATH Level 3 at 512 tokens suffered severe truncation.
- On the same fixed 64 MATH Level 3 prompts, increasing the hard response limit from
  512 to 768 changed:
  - hit-max: 38.09% -> 9.96%;
  - extraction failure: 39.06% -> 11.13%;
  - non-truncation extraction failure: 2.84% -> 1.74%;
  - all-wrong/mixed/all-correct:
    26.56%/54.69%/18.75% -> 12.50%/64.06%/23.44%.
- Therefore `max_response_length=768` is frozen for A–D.
- P90 MATH Level 3 response length was 761, so do not lower the limit back to 512.
- Do not raise it above 768 before A; 768 already passed the calibration criteria.

## 5. Frozen Experiment A protocol

Unless a blocking correctness problem is demonstrated, keep these fixed:

- Model: `Qwen2.5-1.5B-Instruct`
- Algorithm: standard GRPO
- Prompt: zero-shot chat prompt ending with a request to put the final answer in `\boxed{}`
- Group size: `n=8`
- Max response length: `768`
- Reward: binary mathematical correctness
- Training prediction protocol: boxed-only
- Verification backend: the same pinned Math-Verify behavior used by RewardScope
- Initial data mixture: **60% GSM8K + 40% MATH Level 3**
- No format reward
- No length reward
- No Dynamic Sampling
- No Clip-Higher
- No curriculum sampling
- No LoRA unless the research scope is explicitly changed
- No silent CPU fallback

A–D must use the same candidate training pool, prompt, verifier, train/validation split
and evaluation protocol. E may change sampling probability inside the same candidate pool,
but must not silently add new examples.

## 6. Data requirements

Create or reuse veRL-compatible training and validation files.

Each example must retain at least:

- stable `prompt_id`;
- `data_source`: GSM8K or MATH Level 3;
- original source index;
- split;
- chat-format prompt;
- clean ground truth;
- MATH level where applicable.

Required checks before training:

- prompt IDs are unique;
- train and validation have no overlap;
- source ratio is correct;
- every gold answer is parseable by the frozen verifier;
- prompt rendering contains the boxed instruction;
- dataset loading preserves the ground truth and source metadata;
- no parse failure is silently converted into a zero-reward training example;
- the final selected source indices/IDs and selection seed are persisted.

Validation must report GSM8K and MATH Level 3 separately, not only a mixed average.

## 7. Math-Verify integration into veRL

Yes: Experiment A must use Math-Verify inside veRL.

There must be **one source of truth** for reward semantics.

Preferred architecture:

1. Keep the frozen verifier implementation in an importable lightweight RewardScope module,
   or extract only that frozen module into a small shared package.
2. Install it into the veRL environment as a pinned/local package.
3. Add a very thin veRL custom-reward adapter.
4. Do not copy and independently edit the parser/grader logic inside the training repository.
5. The training process must not import RewardScope samplers, reports, plotting or runner code.

Conceptual adapter:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    result = frozen_training_verifier.verify(
        response=solution_str,
        ground_truth=ground_truth,
        data_source=data_source,
    )
    return {
        "score": float(result.is_correct),
        # Adapt the extra-info envelope to this repository's pinned veRL version.
        "reward_extra_info": {
            "raw_correctness": float(result.is_correct),
            "extraction_ok": bool(result.extraction_ok),
            "format_ok": bool(result.format_ok),
            "verification_status": result.status,
        },
    }
```

The exact return envelope varies across veRL versions. Inspect the pinned reward manager and
write against the local version rather than assuming this example is exact.

Required reward-equivalence test:

- Select at least 100 saved RewardScope rollouts across GSM8K and MATH.
- Include correct, incorrect, missing-box, malformed, multi-value and truncated responses.
- Score them through RewardScope and through the veRL adapter.
- Require 100% agreement for correctness, extraction status and format status.
- A parser/library exception must fail loudly and must not be treated as a normal wrong answer.

Measure reward-computation wall time in A0. Math verification should not silently become
the dominant bottleneck.

## 8. A0 smoke-test scope

Start from the existing working 0.5B/1.5B GRPO configuration.

Suggested smoke workload:

- 2–8 prompts per optimizer step;
- `n=8`;
- both GSM8K and MATH Level 3 present;
- 2–3 optimizer steps;
- validation once before training and once after;
- save at least one checkpoint;
- reload the saved checkpoint once.

Keep the real frozen response limit of 768. Do not use a shorter smoke-only limit that
avoids the actual memory shape of Experiment A.

## 9. OOM debugging policy

Do not change the algorithm or research protocol to hide an OOM.

Before running, print and save:

- GPU name and count;
- available VRAM;
- CUDA, PyTorch, veRL, vLLM and Transformers versions;
- model path and dtype;
- resolved train batch size and all micro-batch sizes;
- rollout `n`;
- max prompt and response lengths;
- rollout backend and GPU memory utilization;
- activation checkpointing/offload settings.

If CUDA is unavailable, terminate immediately.

When an OOM occurs, first identify its phase. Apply mitigations in this order, one change
at a time, recording the resolved config after each attempt:

1. For A0 only, reduce prompts per optimizer step.
2. Reduce actor, reference, rollout-logprob and update micro-batches to the smallest
   supported value.
3. Enable gradient/activation checkpointing.
4. Reduce rollout-engine GPU memory utilization or concurrency.
5. Enable parameter/optimizer offload if supported by the pinned veRL path.
6. Confirm model/reference/vLLM coexistence is configured as intended.
7. Only after diagnosis, consider a different sharding/offload strategy.

Do not:

- lower `n` below 8 for the accepted A0;
- lower `max_response_length` below 768 for the accepted A0;
- switch to LoRA;
- change the model to 0.5B and call it the Experiment A smoke test;
- alter the data mixture to make memory fit;
- silently run on CPU.

A smaller pre-A0 environment check may use 0.5B, but it does not count as A0.

## 10. Logs required from A0

At minimum persist:

### Reward and data signal

- reward mean/std;
- all-wrong/mixed/all-correct group rates;
- effective prompt rate;
- extraction failure and format rate;
- metrics split by GSM8K and MATH Level 3;
- response length mean/P90/max;
- hit-max rate;
- generated response-token count.

### Optimization

- policy loss;
- entropy;
- KL;
- clipping fraction;
- gradient norm;
- learning rate;
- advantage mean/std;
- NaN/Inf checks.

### Efficiency and memory

- total step time;
- rollout time;
- reward time;
- reference/log-probability time;
- actor-update time;
- tokens per second;
- peak GPU memory.

Store the resolved configuration, environment/version snapshot and several raw generated
examples with rewards.

## 11. A0 acceptance criteria

A0 passes only if:

1. CUDA is used and no silent CPU fallback occurs.
2. No OOM, NaN, Inf, Ray deadlock or unrecovered worker crash occurs.
3. Every prompt produces exactly eight responses.
4. The veRL reward adapter matches RewardScope on the equivalence fixture.
5. Reward, advantage, loss and gradient values are finite.
6. Both GSM8K and MATH Level 3 enter actual training batches.
7. At least one mixed group is observed.
8. A complete backward and optimizer step succeeds.
9. Validation runs before and after training.
10. A checkpoint saves and reloads successfully.
11. Logs, generated examples and resolved config are persisted.

Do not require accuracy improvement over 2–3 steps.

After A0 passes, freeze the smoke configuration and design the full-budget Experiment A.
Do not begin B before the final A baseline protocol and stopping rule are fixed.

## 12. Fair-comparison rules for A–E

All later variants must share:

- initial checkpoint;
- candidate data pool and train/validation/test split;
- prompt and verifier;
- decoding parameters and group size;
- optimizer and learning-rate policy unless explicitly part of an ablation;
- response-length hard limit;
- evaluation protocol;
- checkpoint-selection rule;
- total rollout-response-token budget as the primary compute budget.

Do not compare only by epoch or optimizer step. Dynamic sampling changes how many generated
groups are accepted, so report generated responses, rollout response tokens, GPU hours and
accepted/effective groups.

Dynamic Sampling grouping must use raw correctness, not shaped final reward.

Keep reward components separate:

- raw correctness;
- format signal;
- length penalty;
- final reward.

Experiment A uses only raw binary correctness.

## 13. Working rules for coding agents

Before modifying code:

1. Read this file.
2. Inspect the repository and identify the existing working execution path.
3. Report the exact files/configs that will be changed.
4. Explain whether each change is required for correctness, memory, observability or
   experiment reproducibility.
5. Prefer minimal changes and small commits.

Never:

- rewrite the trainer from scratch;
- duplicate the verifier;
- modify RewardScope to solve a veRL integration problem;
- add B–E features during A0;
- add a new framework, registry or configuration layer;
- change metric definitions without explicit approval;
- delete legacy code merely because static search finds no direct call;
- change tests to conceal behavioral regressions.

After every meaningful change:

- run targeted tests;
- run the reward-equivalence fixture when reward code changes;
- run `git diff --check`;
- save the exact command and resolved configuration used for GPU runs.

## 14. Immediate next actions

1. Locate and document the previous working 0.5B launch path and the failed 1.5B config/log.
2. Pin the veRL training environment and verify CUDA on the target GPU.
3. Prepare the fixed 60/40 GSM8K/MATH Level 3 train and validation data.
4. Add the thin Math-Verify reward adapter.
5. Build and pass the 100+ rollout reward-equivalence fixture.
6. Run a one-batch no-update or minimal-update integration check if useful.
7. Run the accepted 2–3-step A0 smoke test.
8. Produce an A0 report with the acceptance checklist and memory/timing breakdown.
