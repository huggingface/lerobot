# Processors for Robots and Teleoperators

This guide shows how to build and modify processing pipelines that connect teleoperators (e.g., phone) to robots and datasets. Pipelines standardize conversions between different action/observation spaces so you can swap teleops and robots without rewriting glue code.

We use the Phone to SO‑100 follower examples for concreteness, but the same patterns apply to other robots.

**What you'll learn**

- Absolute vs. relative EE control: What each means, trade‑offs, and how to choose for your task.
- Three-pipeline pattern: How to map teleop actions → dataset actions → robot commands, and robot observations → dataset observations.
- Adapters (`to_transition` / `to_output`): How these convert raw dicts to `EnvTransition` and back to reduce boilerplate.
- Dataset feature contracts: How steps declare features via `transform_features(...)`, and how to aggregate/merge them for recording.
- Choosing a representation: When to store joints, absolute EE poses, or relative EE deltas—and how that affects training.
- Pipeline customization guidance: How to swap robots/URDFs safely and tune bounds, step sizes, and options like IK initialization.

### Absolute vs relative EE control

The examples in this guide use absolute end effector (EE) poses because they are easy to reason about. In practice, relative EE deltas or joint position are often preferred as learning features.

With processors, you choose the learning features you want to use for your policy. This could be joints positions/velocities, absolute EE, or relative EE positions. You can also choose to store other features, such as joint torques, motor currents, etc.

## Three pipelines

We often compose three pipelines. Depending on your setup, some can be empty if action and observation spaces already match.
Each of these pipelines handle different conversions between different action and observation spaces. Below is a quick explanation of each pipeline.

1. Pipeline 1: Teleop action space → dataset action space (phone pose → EE targets)
2. Pipeline 2: Dataset action space → robot command space (EE targets → joints)
3. Pipeline 3: Robot observation space → dataset observation space (joints → EE pose)

Below is an example of the three pipelines that we use in the phone to SO-100 follower examples:

```python
phone_to_robot_ee_pose_processor = RobotProcessorPipeline[RobotAction, RobotAction]( # teleop -> dataset action
    steps=[
        MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
        EEReferenceAndDelta(
            kinematics=kinematics_solver, end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5}, motor_names=list(robot.bus.motors.keys()),
        ),
        EEBoundsAndSafety(
            end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}, max_ee_step_m=0.20,
        ),
        GripperVelocityToJoint(),
    ],
    to_transition=robot_action_to_transition,
    to_output=transition_to_robot_action,
)

robot_ee_to_joints_processor = RobotProcessorPipeline[RobotAction, RobotAction]( # dataset action -> robot
    steps=[
        InverseKinematicsEEToJoints(
            kinematics=kinematics_solver, motor_names=list(robot.bus.motors.keys()), initial_guess_current_joints=True,
        ),
    ],
    to_transition=robot_action_to_transition,
    to_output=transition_to_robot_action,
)

robot_joints_to_ee_pose = RobotProcessorPipeline[RobotObservation, RobotObservation]( # robot obs -> dataset obs
    steps=[
        ForwardKinematicsJointsToEE(kinematics=kinematics_solver, motor_names=list(robot.bus.motors.keys()))
    ],
    to_transition=observation_to_transition,
    to_output=transition_to_observation,
)
```

## Why to_transition / to_output

To convert from robot/teleoperator to pipeline and back, we use the `to_transition` and `to_output` pipeline adapters.
They standardize conversions to reduce boilerplate code, and form the bridge between the robot and teleoperators raw dictionaries and the pipeline’s `EnvTransition` format.
In the phone to SO-100 follower examples we use the following adapters:

- `robot_action_to_transition`: transforms the teleop action dict to a pipeline transition.
- `transition_to_robot_action`: transforms the pipeline transition to a robot action dict.
- `observation_to_transition`: transforms the robot observation dict to a pipeline transition.
- `transition_to_observation`: transforms the pipeline transition to a observation dict.

Checkout [src/lerobot/processor/converters.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/processor/converters.py) for more details.

## Dataset feature contracts

Dataset features are determined by the keys saved in the dataset. Each step can declare what features it modifies in a contract called `transform_features(...)`. Once you build a processor, the processor can then aggregate all of these features with `aggregate_pipeline_dataset_features()` and merge multiple feature dicts with `combine_feature_dicts(...)`.

Below is and example of how we declare features with the `transform_features` method in the phone to SO-100 follower examples:

```python
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We only use the ee pose in the dataset, so we don't need the joint positions
        for n in self.motor_names:
            features[PipelineFeatureType.ACTION].pop(f"{n}.pos", None)
        # We specify the dataset features of this step that we want to be stored in the dataset
        for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION][f"ee.{k}"] = PolicyFeature(
                type=FeatureType.STATE, shape=(1,)
            )
        return features
```

Here we declare what PolicyFeatures we modify in this step, so we know what features we can expect when we run the processor. These features can then be aggregated and used to create the dataset features.

Below is an example of how we aggregate and merge features in the phone to SO-100 record example:

```python
features=combine_feature_dicts(
        # Run the feature contract of the pipelines
        # This tells you how the features would look like after the pipeline steps
        aggregate_pipeline_dataset_features(
            pipeline=phone_to_robot_ee_pose_processor,
            initial_features=create_initial_features(action=phone.action_features), # <- Action features we can expect, these come from our teleop device (phone) and action processor
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_joints_to_ee_pose,
            initial_features=create_initial_features(observation=robot.observation_features), # <- Observation features we can expect, these come from our robot and observation processor
            use_videos=True,
            patterns=["observation.state.ee"], # <- Here you could optionally filter the features we want to store in the dataset, with a specific pattern

        ),
    ),
```

How it works:

- `aggregate_pipeline_dataset_features(...)`: applies `transform_features` across the pipeline and filters by patterns (images included when `use_videos=True`, and state features included when `patterns` is specified).
- `combine_feature_dicts(...)`: combine multiple feature dicts.
- Recording with `record_loop(...)` uses `build_dataset_frame(...)` to build frames consistent with `dataset.features` before we call `add_frame(...)` to add the frame to the dataset.

## Guidance when customizing robot pipelines

You can store any of the following features as your action/observation space:

- Joint positions
- Absolute EE poses
- Relative EE deltas
- Other features: joint velocity, torques, etc.

Pick what you want to use for your policy action and observation space and configure/modify the pipelines and steps accordingly.

### Different robots

- You can easily reuse pipelines, for example to use another robot with phone teleop, modify the examples and swap the robot `RobotKinematics` (URDF) and `motor_names` to use your own robot with Phone teleop. Additionally you should ensure `target_frame_name` points to your gripper/wrist.

### Safety first

- When changing pipelines, start with tight bounds, implement safety steps when working with real robots.
- Its advised to start with simulation first and then move to real robots.

Thats it! We hope this guide helps you get started with customizing your robot pipelines, If you run into any issues at any point, jump into our [Discord community](https://discord.com/invite/s3KuuzsPFb) for support.
