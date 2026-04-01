# Action Representations

This guide explains the different ways robot actions can be represented in LeRobot, how they relate to each other, and when to use each one.

## Joint Space vs End-Effector Space

Before discussing action representations, it helps to understand the two coordinate spaces actions can live in.

### Joint Space

Joint-space actions directly specify target positions for each motor. For a 6-DOF arm with a gripper, a joint-space action might look like:

```
action = [shoulder_pan: 45.0, shoulder_lift: -20.0, elbow: -30.0, wrist_pitch: 10.0, wrist_roll: 0.0, wrist_yaw: 5.0, gripper: 0.8]
```

Joint space is the default in LeRobot. It is simple, requires no kinematics model, and maps directly to motor commands. Most beginner setups (SO-100, Koch) use joint-space actions.

### End-Effector (EE) Space

End-effector-space actions specify the desired position and orientation of the robot's tool tip (gripper) in Cartesian coordinates:

```
action = [x: 0.25, y: -0.10, z: 0.15, wx: 0.0, wy: 0.0, wz: 0.1, gripper: 0.8]
```

EE space is more intuitive for tasks like pick-and-place because it directly describes where the gripper should go, but it requires a kinematics model (URDF) to convert between EE poses and joint angles.

### Converting Between Spaces

LeRobot provides processor steps for converting between joint and EE spaces using forward and inverse kinematics. These are built on top of `RobotKinematics`, which loads a URDF model of your robot.

```python
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower.robot_kinematic_processor import (
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)

kinematics = RobotKinematics(
    urdf_path="./SO101/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=["shoulder", "elbow", "wrist_pitch", "wrist_roll", "wrist_yaw"],
)

# Joints → EE (for observations: "where is my gripper?")
fk_step = ForwardKinematicsJointsToEE(kinematics=kinematics, motor_names=[...])

# EE → Joints (for actions: "move my gripper here")
ik_step = InverseKinematicsEEToJoints(kinematics=kinematics, motor_names=[...])
```

See [`examples/so100_to_so100_EE/`](https://github.com/huggingface/lerobot/tree/main/examples/so100_to_so100_EE) for a complete working example of recording, replaying, and evaluating with EE-space actions on an SO-100 arm.

## Absolute, Relative, and Delta Actions

Regardless of whether you work in joint space or EE space, the action values can be expressed in three different ways. The terminology follows [UMI (Chi et al., 2024)](https://arxiv.org/abs/2402.10329).

### Absolute Actions (LeRobot default)

Each action specifies the target position directly.

**Example** (joint space, chunk of 4):

```
current_state = [45.0, -30.0, 10.0]

action_chunk = [
    [46.0, -29.0, 11.0],   # go to 46, -29, 11
    [47.5, -27.0, 12.0],   # go to 47.5, -27, 12
    [49.0, -25.0, 13.5],   # go to 49, -25, 13.5
    [50.0, -24.0, 15.0],   # go to 50, -24, 15
]
```

Each value is a target position in the robot's coordinate frame. Simple and direct, but requires a consistent global coordinate frame. This is the default in LeRobot.

### Relative Actions (used by OpenPI / pi0)

Each action in the chunk is an offset from the **current state at the moment of prediction**. All actions in the chunk share the same reference point:

```
current_state = [45.0, -30.0, 10.0]

relative_chunk = [
    [1.0,  1.0, 1.0],   # +1 from current → target 46, -29, 11
    [2.5,  3.0, 2.0],   # +2.5 from current → target 47.5, -27, 12
    [4.0,  5.0, 3.5],   # +4 from current → target 49, -25, 13.5
    [5.0,  6.0, 5.0],   # +5 from current → target 50, -24, 15
]
```

The conversion is straightforward: `relative = absolute - current_state`. To recover absolute: `absolute = relative + current_state`.

**Why use relative actions?** The model learns to predict offsets centered around zero, which is easier to normalize and leads to more stable training. Because every chunk references the same current state, there is no error accumulation across chunks.

### Delta Actions (sequential differences)

Each action is an offset from the **previous action** (or from the current state for the first step):

```
current_state = [45.0, -30.0, 10.0]

delta_chunk = [
    [1.0,  1.0, 1.0],   # current → 46, -29, 11
    [1.5,  2.0, 1.0],   # previous action → 47.5, -27, 12
    [1.5,  2.0, 1.5],   # previous action → 49, -25, 13.5
    [1.0,  1.0, 1.5],   # previous action → 50, -24, 15
]
```

Here each step is relative to the one before it. To recover absolute positions you must sum all previous deltas, which means errors accumulate over time. UMI explicitly argues against this representation for this reason.

### Visual Comparison

The figure below (based on a figure from [UMI, Chi et al., 2024](https://arxiv.org/abs/2402.10329)) illustrates the key difference. With **relative trajectory**, every action in the chunk points back to the same origin (current state), so a new inference step cleanly resets the reference. With **delta**, each action depends on the previous one, so errors accumulate. **Absolute** actions require a consistent global coordinate frame.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/action_representations_umi.png"
  alt="Relative Trajectory as Action Representation (UMI, Chi et al., 2024)"
  width="85%"
/>

## Using Relative Actions in LeRobot

LeRobot provides `RelativeActionsProcessorStep` to convert between absolute and relative actions inside the processor pipeline. This is how pi0, pi0.5, and pi0_fast support relative actions.

> **Note:** All pi models (pi0, pi0.5, pi0*fast) apply relative conversion \_before* normalization (`relative → normalize`), so the normalizer always sees delta (relative) values. This means **relative action stats are required** for all of them when training with `use_relative_actions=true`. In pi0_fast the `RelativeActionsProcessorStep` only modifies the action — the state observation is unchanged — so `NormalizerProcessorStep` still runs before the state tokenizer and the tokenizer continues to receive normalized state as expected.

### How it works

During **training** (preprocessing), actions are converted from absolute to relative before the model sees them:

```
raw absolute action → RelativeActionsProcessorStep → normalize → model
```

During **inference** (postprocessing), model predictions are converted back to absolute before being sent to the robot:

```
model output → unnormalize → AbsoluteActionsProcessorStep → robot
```

The `AbsoluteActionsProcessorStep` reads the cached current state from its paired `RelativeActionsProcessorStep`, so the two must be wired together (handled automatically by the policy factory).

### Enabling relative actions for the pi family (pi0, pi0.5, pi0_fast)

**Step 1**: Precompute relative action statistics for your dataset:

```bash
lerobot-edit-dataset \
    --repo_id your_dataset \
    --operation.type recompute_stats \
    --operation.relative_action true \
    --operation.chunk_size 50 \
    --operation.relative_exclude_joints "['gripper']"
```

**Step 2**: Train with relative actions enabled:

```bash
lerobot-train \
    --dataset.repo_id=your_dataset \
    --policy.type=pi0 \
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints='["gripper"]'
```

The `relative_exclude_joints` parameter specifies joints that should remain in absolute space. For example, gripper commands are typically binary (open/close) and don't benefit from relative encoding.

### Combining relative actions with RTC

[RTC](https://arxiv.org/abs/2506.07339) runs policy inference at high frequency and sends actions to the robot as they are predicted rather than waiting for a full chunk. Relative actions and RTC are fully compatible: because every chunk in relative mode references the **same** current state (captured at the start of inference), each predicted action in the chunk remains a valid offset even if the robot has already moved. No special handling is needed — `RelativeActionsProcessorStep` caches the state once per inference call and `AbsoluteActionsProcessorStep` applies it to every action in the streamed output.

### Combining relative actions with EE space

Relative actions work in both joint space and EE space. For example, if your dataset stores EE actions, relative encoding converts them to offsets from the current EE pose:

```
current_ee_state = [x: 0.25, y: -0.10, z: 0.15, gripper: 0.8]

absolute_ee_chunk = [
    [0.26, -0.09, 0.16, 0.8],
    [0.28, -0.07, 0.18, 0.8],
]

relative_ee_chunk = [
    [0.01,  0.01, 0.01, 0.0],   # offset from current EE pose
    [0.03,  0.03, 0.03, 0.0],   # offset from current EE pose
]
```

## Processing Pipeline Summary

Here is how the different processors compose. Each arrow is a processor step, and they can be chained in a `RobotProcessorPipeline` or `PolicyProcessorPipeline`:

```
                    ┌─────────────────────────────────────────┐
   Action Space     │   Joint Space  ←──IK──→  EE Space      │
                    │   ForwardKinematicsJointsToEE           │
                    │   InverseKinematicsEEToJoints           │
                    └─────────────────────────────────────────┘

                    ┌─────────────────────────────────────────┐
   Representation   │   Absolute  ←────→  Relative            │
                    │   RelativeActionsProcessorStep (pre)    │
                    │   AbsoluteActionsProcessorStep (post)   │
                    └─────────────────────────────────────────┘

                    ┌─────────────────────────────────────────┐
   Normalization    │   Raw  ←────→  Normalized               │
                    │   NormalizerProcessorStep (pre)         │
                    │   UnnormalizerProcessorStep (post)      │
                    └─────────────────────────────────────────┘
```

A typical training preprocessor might chain: `raw absolute joint actions → relative → normalize`. A typical inference postprocessor: `unnormalize → absolute → (optionally IK to joints)`.

## References

- [Universal Manipulation Interface (UMI)](https://arxiv.org/abs/2402.10329) - Chi et al., 2024. Defines the relative trajectory action representation and compares it with absolute and delta actions.
- [Introduction to Processors](./introduction_processors) - How processor pipelines work in LeRobot.
- [`examples/so100_to_so100_EE/`](https://github.com/huggingface/lerobot/tree/main/examples/so100_to_so100_EE) - Complete example of recording and evaluating with EE-space actions.
