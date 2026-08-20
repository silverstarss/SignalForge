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
