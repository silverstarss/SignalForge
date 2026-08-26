# HIVE Phase 8 Metrics Audit

Date: 2026-08-26

Scope: `HIVE_IMPLEMENTATION_SPEC.md` Sections 18 and 19. This audit records the
state found before the Phase 8 metrics patch and the final state after the
minimum analysis-required additions. It does not change HIVE selection,
rollout, reward, top-up, history, or GRPO behavior.

Status meanings:

- `PASS`: the required value was already complete and semantically correct.
- `MISSING`: the required metric key/value was not emitted.
- `INCORRECT`: a related metric existed, but its scope or accounting did not
  satisfy the specification.
- `N/A`: explicitly conditional in the specification and not enabled in the
  faithful `lambda=1.0` main path.

## Section 18 Audit

| Requirement | Before Phase 8 | Finding | Final |
|---|---|---|---|
| `hive/raw_prompts_seen`, `unseen_prompts_seen` | INCORRECT | Initial acquisition only; top-up acquisitions were absent from the main per-step value. | PASS |
| Stage-1 accepted/rejected/ratio | INCORRECT | Initial acquisition only. | PASS |
| Stage-2 input/kept/upper/low counts | INCORRECT | Initial acquisition only. | PASS |
| `hive/stage2_keep_ratio` | MISSING | No exact Section 18 ratio. | PASS |
| Entropy mean and q25/q50/q75 | INCORRECT | Present, but initial acquisition only. | PASS |
| Entropy std/min/max | MISSING | Not emitted. | PASS |
| Selected/rejected entropy means | MISSING | Not emitted. | PASS |
| Historical response entropy/correlation | N/A | Section 18 requires these only when historical response entropy is implemented; main reproduction uses `lambda=1.0` and no entropy proxy. | N/A |
| easy/hard/other/total zero-var counts | PASS | Cumulative over every rollout and top-up round in the optimizer step. | PASS |
| easy/hard/total zero-var ratios | PASS | Denominator is every complete prompt group generated in the step. | PASS |
| `hive/p_easy`, `hive/p_hard` | MISSING | Before/after keys existed, but the exact Section 18 aliases did not. | PASS |
| Six cumulative generated/effective compute counters | INCORRECT | HIVE had the exact keys; baseline A had separate `budget/*` aliases, and the common tracker was not checkpointed. | PASS |
| Four derived efficiency metrics | MISSING | Ratios and groups-per-1M-token value were absent. | PASS |
| Stage-1 timing | MISSING | No dedicated timer. | PASS |
| Stage-2 entropy timing | MISSING | Actor RPC latency existed, but the exact `time/stage2_entropy` key did not. | PASS |
| Rollout/reward timing | INCORRECT | Native timers excluded top-up rollout/reward work. | PASS |
| Top-up/update/validation/checkpoint timing | MISSING | Native timers existed, but the exact Section 18 keys were absent. | PASS |
| Iteration total timing | INCORRECT | The timer began after HIVE pre-rollout selection. | PASS |
| Top-up rounds/target/effective before/after | PASS | Existing aggregate top-up metrics were correct. | PASS |
| `hive/estimated_zero_var_ratio` | MISSING | Equivalent `hive/topup_rho_zv` existed without the required stable alias. | PASS |

## Metrics Added

The exact added stable keys are:

```text
hive/stage2_keep_ratio
hive/prompt_entropy_std
hive/prompt_entropy_min
hive/prompt_entropy_max
hive/selected_entropy_mean
hive/rejected_entropy_mean
hive/p_easy
hive/p_hard
hive/estimated_zero_var_ratio

compute/effective_prompt_group_ratio
compute/effective_response_ratio
compute/effective_training_token_ratio
compute/effective_prompt_groups_per_1m_generated_response_tokens

time/stage1
time/stage2_entropy
time/rollout
time/reward
time/topup
time/grpo_update
time/validation
time/checkpoint
time/iteration_total
time/rollout_wall_clock_cumulative
time/total_wall_clock_cumulative
```

The six Section 18 cumulative base keys are now emitted through one common
tracker for both formal A and B:

```text
compute/generated_prompt_groups
compute/generated_responses
compute/generated_response_tokens
compute/effective_prompt_groups
compute/effective_responses
compute/effective_response_tokens
```

## A/B Compute Parity

Both baseline A and HIVE B now update and checkpoint the same
`RolloutBudgetTracker`.

- Generated counts are measured before filtering and include discarded
  rollouts.
- Effective counts are measured after the relevant filtering boundary.
- In formal baseline A, where Dynamic Sampling is disabled, effective counts
  equal generated counts.
- In HIVE B, effective counts include all non-zero-variance groups, including
  effective overshoot groups not selected into the final `B_t` training
  slice.
- Response-token counts use the same `response_mask` accounting in A and B.
- At every HIVE step, the six common counters are checked against
  `HiveComputeCounters`; disagreement is fatal.
- The common tracker is checkpointed with optimizer-step validation. For a
  legacy HIVE checkpoint, the recoverable six counters are migrated from
  `HiveComputeCounters`; historical prompt-token and wall-clock values,
  which do not exist in the legacy checkpoint, restart at the resume boundary
  with an explicit warning.

Here, `effective_training_token_ratio` uses effective response tokens divided
by generated response tokens, matching the Section 18 generated-token budget.

## Section 19 Curves

The trainer already emits `training/global_step` and validation
`val/pass_at_1` (plus per-source validation aliases). Each logged validation
point, including the pre-training validation point, is logged with the cumulative metrics below:

| Required curve axis | Logged value |
|---|---|
| training step | `training/global_step` |
| generated responses | `compute/generated_responses` |
| generated response tokens | `compute/generated_response_tokens` |
| rollout wall-clock | `time/rollout_wall_clock_cumulative` |
| total wall-clock | `time/total_wall_clock_cumulative` |

The remaining mechanism curves use the per-step keys required by Section 18:
effective prompt ratio, total zero-var ratio, `p_easy`/`p_hard`, and prompt
entropy quantiles.

No new validation metric, HIVE decision rule, or trainer control-flow
abstraction was introduced in Phase 8.
