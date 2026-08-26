# HIVE Deviations and Ambiguity Resolutions

This file records reviewed choices that cannot be derived unambiguously from the HIVE paper.
It does not override `docs/hive/HIVE_IMPLEMENTATION_SPEC.md` except where a reviewed resolution is stated.

## HIVE-001: Exploration for other zero-variance groups

- **Status:** Approved on 2026-08-19; Phase 1 stores the configuration and Phase 2 implements the fixed default.
- **Specification references:** Sections 6.3 and 15.6.
- **Ambiguity:** The paper defines easy zero-variance groups at reward `1.0` and hard zero-variance groups at
  reward `0.1`, but does not define the exploration controller for another constant reward group such as
  extraction failures at `0.0`.
- **Resolution:** Classify every zero-variance group that is neither all `1.0` nor all `0.1` as `other`.
  Use a configurable fixed `p_default`, with default value `0.5`, when Stage 1 is implemented. Do not update
  `p_default` through the easy/hard adaptive controller.
- **Reason:** This keeps the unspecified class explicit and avoids silently mapping extraction failures to
  either paper-defined difficulty class.
- **Evaluation impact:** Report `other` zero-variance frequency separately. Any experiment that changes
  `p_default` is an ablation and must record the value.

## HIVE-002: Extraction-failure reward

- **Status:** Approved on 2026-08-19; classification is implemented in Phase 1, reward-adapter integration is
  deferred.
- **Specification references:** Sections 6.3, 15.6, and 16.
- **Ambiguity:** The paper explicitly defines rewards for correct and extractable-but-incorrect responses but
  does not fully specify the reward for extraction failure.
- **Resolution:** Use the following reward semantics for the HIVE reproduction:

  ```text
  correct                    -> 1.0
  extractable but incorrect  -> 0.1
  extraction failure         -> 0.0
  ```

  Consequently, all-`1.0` groups are `easy`, all-`0.1` groups are `hard`, and all-`0.0` groups are `other`.
- **Reason:** A separate `0.0` value preserves the distinction between an extracted wrong answer and a failed
  extraction.

## HIVE-003: Initial candidate pool ratio

- **Status:** Approved on 2026-08-20; Phase 5C pre-rollout accumulation is implemented.
- **Specification references:** Sections 10, 15.2, and 15.7.
- **Ambiguity:** Appendix B.3 and Algorithm 1 specify `B_cand = 1.5 * B_t`, while Appendix F prose describes
  a `2 * B_t` pool.
- **Resolution:** Use the Appendix B.3 deployed rule `B_cand_target = 3 * B_t / 2`, require the result to be an
  integer, and retain complete per-`b_raw` Stage-2-kept partitions until the target lower bound is reached.
- **Reason:** Appendix B.3 is the approved implementation source for the faithful main reproduction.
- **Evaluation impact:** Log actual candidate count, overshoot, actual ratio to `B_t`, and accumulation rounds; do
  not compensate for discrete `G`-multiple overshoot.

## HIVE-004: Zero-survival adaptive top-up boundary

- **Status:** Approved on 2026-08-21.
- **Specification references:** Sections 13, 13.1, and 15.
- **Ambiguity:** The paper adaptive top-up equation is undefined at exact
  `rho_zv = 1` because its survival denominator is zero.
- **Resolution:** If effective groups are still required and
  `1 - rho_zv <= survival_epsilon`, set `B_cand_adapt = B_cand` and
  `candidate_cap_binding = true`, then perform a normal complete top-up
  round. Do not divide by an artificial epsilon or add pseudocounts.
- **Reason:** The left-limit of the published formula, after its explicit
  candidate cap is applied, saturates at `B_cand`.
- **Evaluation impact:** Zero-survival rounds remain fully subject to the
  existing filtering semantics. Repeated zero-survival rounds terminate
  through `max_topup_rounds` and/or dataloader exhaustion.

## HIVE-005: Qwen2.5-3B model-relative dataset mixture

- **Status:** Approved on 2026-08-22 after the 256 MATH + 256 DAPO calibration.
- **Specification references:** Sections 20, 21, and 22.
- **Adaptation:** Use a prompt-level mixture of 75% MATH and 25% DAPO-Math for
  the current Qwen2.5-3B-Instruct reproduction candidate. This ratio is a
  model-relative dataset adaptation, not a claim that the paper prescribes a
  universal 3:1 mixture.
- **Sources:** `EleutherAI/hendrycks_math` train revision
  `21a5633873b6a120296cce3e2df9d5550074f4a3` and
  `BytedTsinghua-SIA/DAPO-Math-17k` train revision
  `65877096c24ffa7abc4e4fa5edb95cf3413a5674`.
- **Preprocessing:** Normalize both sources to the single canonical boxed-answer
  prompt recorded in `EXPERIMENT_PROTOCOL.md`. Deduplicate DAPO by
  `extra_info.index`, reject conflicting duplicates, and validate normalized
  ground truth with the frozen LaTeX verifier.
- **Identity:** Preserve dataset-qualified stable IDs in every mixture:
  `math:<source_row_id>` and `dapo:<extra_info.index>`.
- **Frozen clean pool:** Keep all validation rows, remove confirmed `A` exact
  and `B` trivial-paraphrase overlaps from the complete training source pools,
  then recompute the maximal exact 3:1 ratio. The frozen result is 7,488 MATH
  plus 2,496 DAPO prompts (9,984 total); the decisions and source-qualified IDs
  are recorded under
  `artifacts/formal_data/hive_math75_dapo25_seed42_validation_clean_max_exact_3to1`.
- **Reason:** At `G=8`, temperature `1.0`, and calibration response limit
  `1536`, MATH had a materially thicker 1/8--7/8 band and a smaller hard tail;
  DAPO supplied harder examples but was dominated by 0/8 groups. The reviewed
  3:1 projection retained DAPO diversity while reducing the hard tail and
  generated-token cost relative to 1:1 and 3:2 candidates.
- **Evaluation impact:** Report source composition, the complete correct-count
  histogram, easy/hard/other/effective rates, truncation, extraction failures,
  and generated tokens. Do not revise the mixture from effective ratio alone.

## HIVE-006: Single-GPU adaptive top-up minimum

- **Status:** Approved and frozen on 2026-08-25 after the 80-step single-GPU
  HIVE mechanism pilot.
- **Specification references:** Sections 10 and 13.
- **Adaptation:** Use `b_min = 8` for the current formal reproduction on one
  RTX PRO 6000D, with `B_t = 32` and the faithful derived
  `B_cand = 3 * B_t / 2 = 48`. This replaces the paper-scale default
  `b_min = 64` while preserving the required `b_min <= B_cand` domain.
- **Reason:** The paper uses `b_min = 64` with `B_cand = 192` or `384`; using
  `64` with the single-GPU `B_cand = 48` would violate the approved preflight
  invariant. In the reviewed 80-step pilot, top-up was triggered in 29 steps,
  `b_min` was binding in 7 of those steps, every optimizer step obtained the
  exact `B_t = 32` complete effective groups, and no step required more than
  one top-up round.
- **Algorithmic impact:** None beyond the explicitly scaled lower bound. The
  published adaptive equation, candidate cap, frozen selector semantics,
  complete-partition overshoot rule, and zero-variance filtering are unchanged.
- **Evaluation impact:** Report `b_min`, `b_min_binding`, adaptive candidate
  targets, actual candidates, overshoot, and top-up rounds. Do not present
  `b_min = 8` as a paper-default value or generalize it beyond this single-GPU
  reproduction.
