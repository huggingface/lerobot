# OpenArm Gripette Manual Policy Start Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the operator place the OpenArm manually before policy execution, then enable torque and hold that exact measured pose.

**Architecture:** Add one small hardware-preparation helper to the third-party plugin. The policy runner calls it after connecting, uses its returned URDF-frame joint positions as the FK/IK state, prints the held Cartesian pose, and retains its existing final motion confirmation.

**Tech Stack:** Python 3.12, NumPy, LeRobot `OpenArmGripetteFollower`, pytest mocks.

## Global Constraints

- Leave Gripette torque and streaming enabled during manual arm placement.
- Send arm torque enable/disable twice with a 50 ms pause between passes.
- Do not automatically move to a fixed home pose.
- Do not push commits.

---

### Task 1: Manual placement and hold

**Files:**
- Create: `lerobot_robot_openarm_gripette/lerobot_robot_openarm_gripette/manual_start.py`
- Modify: `lerobot_robot_openarm_gripette/examples/run_grabette_policy.py`
- Modify: `lerobot_robot_openarm_gripette/tests/test_openarm_gripette_follower.py`
- Modify: `lerobot_robot_openarm_gripette/README.md`

**Interfaces:**
- Consumes: connected `OpenArmGripetteFollower`, seven robot-to-URDF joint signs, injectable prompt and sleep callables.
- Produces: `prepare_manual_start(robot, signs, input_fn=input, sleep_fn=time.sleep) -> np.ndarray`, containing seven held URDF-frame joint positions in degrees.

- [ ] **Step 1: Write the failing test**

Add a grouped test which mocks `robot.arm.bus.disable_torque`, `enable_torque`, `get_observation`, and `send_action`. Verify the exact call order:

```python
events = []
robot.arm.bus.disable_torque.side_effect = lambda: events.append("disable")
robot.arm.bus.enable_torque.side_effect = lambda: events.append("enable")
robot.arm.get_observation.return_value = {
    f"joint_{i}.pos": value for i, value in enumerate([1, 2, 3, 4, 5, 6, 7], start=1)
}

result = prepare_manual_start(
    robot,
    np.array([1, -1, 1, -1, 1, -1, 1]),
    input_fn=lambda _: events.append("prompt"),
    sleep_fn=lambda _: events.append("sleep"),
)

assert events == ["disable", "sleep", "disable", "prompt", "enable", "sleep", "enable"]
assert result.tolist() == [1, -2, 3, -4, 5, -6, 7]
robot.arm.send_action.assert_called_once_with(
    {f"joint_{i}.pos": float(i) for i in range(1, 8)}
)
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
HF_LEROBOT_CALIBRATION=/tmp/lerobot_calib \
PYTHONPATH=src:lerobot_robot_openarm_gripette \
conda run -n lerobot-dev python -m pytest \
  lerobot_robot_openarm_gripette/tests/test_openarm_gripette_follower.py -q -p no:cacheprovider
```

Expected: import failure for `lerobot_robot_openarm_gripette.manual_start`.

- [ ] **Step 3: Implement the helper**

Create:

```python
def prepare_manual_start(
    robot: OpenArmGripetteFollower,
    signs: np.ndarray,
    input_fn: Callable[[str], str | None] = input,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> np.ndarray:
    if signs.shape != (7,) or not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("signs must contain seven values, each +1 or -1")

    robot.arm.bus.disable_torque()
    sleep_fn(0.05)
    robot.arm.bus.disable_torque()
    input_fn("Arm torque is OFF. Support and place the arm, then press ENTER...")

    observation = robot.arm.get_observation()
    measured = np.array([observation[f"joint_{i}.pos"] for i in range(1, 8)], dtype=np.float64)

    robot.arm.bus.enable_torque()
    sleep_fn(0.05)
    robot.arm.bus.enable_torque()
    robot.arm.send_action({f"joint_{i}.pos": float(value) for i, value in enumerate(measured, start=1)})
    return signs * measured
```

- [ ] **Step 4: Wire it into policy startup**

After `robot.connect()`, replace the initial measured-joint assignment with:

```python
q_cmd = prepare_manual_start(robot, signs)
held = robot.arm.get_observation()
q_cmd = signs * np.array([held[f"{joint}.pos"] for joint in ROBOT_JOINTS])
print("Held joints (URDF deg):", np.round(q_cmd, 1))
print("Held oak_l position (m):", np.round(kin.forward_kinematics(q_cmd)[:3, 3], 3))
input(f"{'DRY RUN: the arm will not move. ' if args.dry_run else ''}Press ENTER to start...")
```

Remove the previous single start-position prompt.

- [ ] **Step 5: Document the startup sequence**

Document that the policy runner disables only arm torque, waits for manual placement, holds the measured pose on ENTER, prints it, and asks for a second confirmation before policy execution.

- [ ] **Step 6: Verify**

Run:

```bash
conda run -n lerobot-dev ruff check lerobot_robot_openarm_gripette
conda run -n lerobot-dev ruff format --check lerobot_robot_openarm_gripette
MYPYPATH=src conda run -n lerobot-dev mypy --ignore-missing-imports \
  lerobot_robot_openarm_gripette/lerobot_robot_openarm_gripette
HF_LEROBOT_CALIBRATION=/tmp/lerobot_calib \
PYTHONPATH=src:lerobot_robot_openarm_gripette \
conda run -n lerobot-dev python -m pytest \
  lerobot_robot_openarm_gripette/tests -q -p no:cacheprovider
```

Expected: all commands pass.

- [ ] **Step 7: Commit**

```bash
git add lerobot_robot_openarm_gripette docs/superpowers/plans/2026-10-08-openarm-gripette-manual-start.md
git commit -m "feat(openarm-gripette): add manual policy start positioning"
```
