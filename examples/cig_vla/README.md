# CIG-VLA: LIBERO-Safety trajectory supervision

CIG-VLA uses the public `LIBERO-Safety/libero_safety` v2.1 schema directly. It predicts robot-centric interaction geometry from two RGB cameras, instruction, and robot state; no object pose, contact phase, frame-level safety label, or simulator-generated label is used. The demonstrations are collision-free behavior-cloning data, not unsafe-counterexample supervision.

The lazy reader downloads metadata once and caches only requested per-episode parquet/video files through the Hugging Face cache. It maps `actions` to `action`, resolves `task_index` to instruction, and pads chunks without crossing episode boundaries.

Run contract and correctness checks:

```bash
uv run python examples/cig_vla/inspect_libero_safety.py
uv run pytest tests/policies/cig_vla tests/datasets/test_libero_safety_adapter.py tests/examples tests/scripts/test_collect_cig_geometry_labels.py -q
uv run python examples/cig_vla/overfit_cig_grounding.py --mock-backbone
uv run python examples/cig_vla/overfit_cig_controller.py
uv run python examples/cig_vla/smoke_train_cig_vla.py --mock-backbone
```

Integration smoke (this is not a pilot):

```bash
uv run python examples/cig_vla/libero_safety_smoke.py --steps 1 --output-dir /tmp/cig_libero_a1
uv run python examples/cig_vla/libero_safety_smoke.py --steps 1 --local-response --output-dir /tmp/cig_libero_a3
uv run python examples/cig_vla/check_libero_safety_readiness.py --run-actual-qwen
```

The strict readiness command exits non-zero unless actual Qwen one-batch backward, 10/10 updates, checkpoint reload, and fixed-input inference all pass. Do not run the pilot configs until it prints `READY FOR A0/A1/A3 PILOT`.

Pilot configs, gated by that verdict (use the same seed and task subset for all runs):

```bash
uv run lerobot-train --config_path=examples/cig_vla/configs/a0_direct.yaml --dataset.repo_id=LIBERO-Safety/libero_safety
uv run lerobot-train --config_path=examples/cig_vla/configs/a1_interaction.yaml --dataset.repo_id=LIBERO-Safety/libero_safety
uv run lerobot-train --config_path=examples/cig_vla/configs/a3_local_response.yaml --dataset.repo_id=LIBERO-Safety/libero_safety
```

A0 is an implicit-bottleneck, no-explicit-geometry-supervision baseline that preserves the same model footprint; it is not a fully monolithic Qwen-to-action implementation. A1 adds trajectory-derived geometry supervision. A3 adds a low-weight, same-sign local translation-response regularizer; it does not invent counterfactual target actions.

Online success and safety metrics require the official LIBERO-Safety benchmark environment/evaluator. Pass that evaluator to `LiberoSafetyOnlineBackend`; do not infer safety-violation rate from the offline demonstrations. Offline evaluation is limited to imitation error, translation goal error, direction cosine error, gripper-transition accuracy, intervention response, and removal sensitivity.
