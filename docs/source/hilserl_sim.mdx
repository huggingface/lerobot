# Train RL in Simulation

This guide explains how to use the `gym_hil` simulation environments as an alternative to real robots when working with the LeRobot framework for Human-In-the-Loop (HIL) reinforcement learning.

`gym_hil` is a package that provides Gymnasium-compatible simulation environments specifically designed for Human-In-the-Loop reinforcement learning. These environments allow you to:

- Train policies in simulation to test the RL stack before training on real robots

- Collect demonstrations in sim using external devices like gamepads or keyboards
- Perform human interventions during policy learning

Currently, the main environment is a Franka Panda robot simulation based on MuJoCo, with tasks like picking up a cube.

## Installation

First, install the `gym_hil` package within the LeRobot environment:

```bash
pip install -e ".[hilserl]"
```

## What do I need?

- A gamepad or keyboard to control the robot
- A Nvidia GPU

## Configuration

To use `gym_hil` with LeRobot, you need to create a configuration file. An example is provided [here](https://huggingface.co/datasets/lerobot/config_examples/resolve/main/rl/gym_hil/env_config.json). Key configuration sections include:

### Environment Type and Task

```json
{
  "env": {
    "type": "gym_manipulator",
    "name": "gym_hil",
    "task": "PandaPickCubeGamepad-v0",
    "fps": 10
  },
  "device": "cuda"
}
```

Available tasks:

- `PandaPickCubeBase-v0`: Basic environment
- `PandaPickCubeGamepad-v0`: With gamepad control
- `PandaPickCubeKeyboard-v0`: With keyboard control

### Processor Configuration

```json
{
  "env": {
    "processor": {
      "control_mode": "gamepad",
      "gripper": {
        "use_gripper": true,
        "gripper_penalty": -0.02
      },
      "reset": {
        "control_time_s": 15.0,
        "fixed_reset_joint_positions": [
          0.0, 0.195, 0.0, -2.43, 0.0, 2.62, 0.785
        ]
      },
      "inverse_kinematics": {
        "end_effector_step_sizes": {
          "x": 0.025,
          "y": 0.025,
          "z": 0.025
        }
      }
    }
  }
}
```

Important parameters:

- `gripper.gripper_penalty`: Penalty for excessive gripper movement
- `gripper.use_gripper`: Whether to enable gripper control
- `inverse_kinematics.end_effector_step_sizes`: Size of the steps in the x,y,z axes of the end-effector
- `control_mode`: Set to `"gamepad"` to use a gamepad controller

## Running with HIL RL of LeRobot

### Basic Usage

To run the environment, set mode to null:

```bash
python -m lerobot.rl.gym_manipulator --config_path path/to/gym_hil_env.json
```

### Recording a Dataset

To collect a dataset, set the mode to `record` whilst defining the repo_id and number of episodes to record:

```json
{
  "env": {
    "type": "gym_manipulator",
    "name": "gym_hil",
    "task": "PandaPickCubeGamepad-v0"
  },
  "dataset": {
    "repo_id": "username/sim_dataset",
    "root": null,
    "task": "pick_cube",
    "num_episodes_to_record": 10,
    "replay_episode": null,
    "push_to_hub": true
  },
  "mode": "record"
}
```

```bash
python -m lerobot.rl.gym_manipulator --config_path path/to/gym_hil_env.json
```

### Training a Policy

To train a policy, checkout the configuration example available [here](https://huggingface.co/datasets/lerobot/config_examples/resolve/main/rl/gym_hil/train_config.json) and run the actor and learner servers:

```bash
python -m lerobot.rl.actor --config_path path/to/train_gym_hil_env.json
```

In a different terminal, run the learner server:

```bash
python -m lerobot.rl.learner --config_path path/to/train_gym_hil_env.json
```

The simulation environment provides a safe and repeatable way to develop and test your Human-In-the-Loop reinforcement learning components before deploying to real robots.

Congrats 🎉, you have finished this tutorial!

> [!TIP]
> If you have any questions or need help, please reach out on [Discord](https://discord.com/invite/s3KuuzsPFb).

Paper citation:

```
@article{luo2024precise,
  title={Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning},
  author={Luo, Jianlan and Xu, Charles and Wu, Jeffrey and Levine, Sergey},
  journal={arXiv preprint arXiv:2410.21845},
  year={2024}
}
```
