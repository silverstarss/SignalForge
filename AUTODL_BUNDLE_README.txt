Signal Forge A0 AutoDL bundle
created_at=20260730_145001

Extract on AutoDL, then run:
  bash src/scripts_a0/12_autodl_nogpu_setup.sh

Recommended no-GPU checks:
  bash /workspace/src/scripts_a0/00_prepare_a0_data.sh
  bash /workspace/src/scripts_a0/01_check_reward_equivalence.sh
  bash /workspace/src/scripts_a0/02_check_verl_reward_manager.sh

4090 0.5B regression:
  bash /workspace/src/scripts_a0/05_unpack_qwen25_0p5b_model.sh
  bash /workspace/src/scripts_a0/06_prepare_a0_0p5b_regression_data.sh
  bash /workspace/src/scripts_a0/07_run_a0_0p5b_short.sh
  bash /workspace/src/scripts_a0/08_run_a0_0p5b_regression.sh
  bash /workspace/src/scripts_a0/09_reload_a0_0p5b_regression_checkpoint.sh
