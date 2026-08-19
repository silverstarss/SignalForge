# HIVE Deviations and Ambiguity Resolutions

This file records reviewed choices that cannot be derived unambiguously from the HIVE paper.
It does not override `docs/hive/HIVE_IMPLEMENTATION_SPEC.md` except where a reviewed resolution is stated.

## HIVE-001: Exploration for other zero-variance groups

- **Status:** Approved on 2026-08-19; Phase 1 stores the configuration, Stage 1 behavior is not implemented yet.
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
