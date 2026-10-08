# OpenArm Gripette Policy Prediction Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save raw policy predictions and derived robot targets in a crash-resilient JSONL file.

**Architecture:** A small `PredictionLogger` owns file creation and JSON serialization. The policy runner records metadata after manual positioning and records each action after IK and gripper decoding.

**Tech Stack:** Python 3.12 standard library (`json`, `datetime`, `pathlib`), NumPy, pytest.

## Global Constraints

- Default log directory: `outputs/openarm_gripette_policy`.
- Flush after every JSONL record.
- Log dry-run predictions without marking them executed.
- Do not push commits.

---

### Task 1: Prediction logger and rollout integration

**Files:**
- Create: `lerobot_robot_openarm_gripette/lerobot_robot_openarm_gripette/prediction_logger.py`
- Modify: `lerobot_robot_openarm_gripette/examples/run_grabette_policy.py`
- Modify: `lerobot_robot_openarm_gripette/tests/test_openarm_gripette_follower.py`
- Modify: `lerobot_robot_openarm_gripette/README.md`

**Interfaces:**
- Produces: `PredictionLogger(log_dir: Path)`, `write(record: dict)`, `close()`, and `path`.

- [ ] **Step 1: Write a failing test**

Write metadata and one action containing NumPy values, close the logger, parse the JSONL file,
and assert two records are present with the expected values.

- [ ] **Step 2: Verify the test fails**

Run the plugin test file and expect an import error for `prediction_logger`.

- [ ] **Step 3: Implement `PredictionLogger`**

Create the directory, open a UTC timestamped JSONL file, recursively convert NumPy scalars and
arrays through `json.dumps(..., default=...)`, append one line per call, and flush immediately.

- [ ] **Step 4: Integrate into the rollout**

Add `--log_dir`. Open the logger before connecting hardware, write metadata after the held start
pose is known, and log every action after IK and grasp projection. Close it in the outer `finally`.

- [ ] **Step 5: Document the output**

Add the default path and one example record description to the plugin README.

- [ ] **Step 6: Verify and commit**

Run ruff, formatting, mypy, all plugin tests, and the offline policy/IK smoke. Commit as:

```bash
git commit -m "feat(openarm-gripette): log policy predictions"
```
