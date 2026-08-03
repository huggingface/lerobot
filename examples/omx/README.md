# OMX Follower — Cube Pick And Place Example

This is an example of what is possible to do with LeRobot on a physical setup.
It is a WIP and being used internally at LeRobot and specific to our setup, but we hope it can be a useful reference for how to use LeRobot APIs and CLIs.

It includes an end-to-end example for the **OMX Follower** robot arm: pick and place a cube dataset, train a policy, and deploy it autonomously.

## Hardware

| Component | Value                                |
| --------- | ------------------------------------ |
| Robot     | OMX Follower                         |
| Cameras   | 2× OpenCV cameras (wrist + top-down) |

## Scripts

| Script                 | Purpose                                                         |
| ---------------------- | --------------------------------------------------------------- |
| `reset_environment.py` | Standalone utility: sweep workspace, grab cube, place cube      |
| `record_grab.py`       | Automated data collection: reset → place → record grab episodes |

## Setup

Make sure you have LeRobot installed in your env. (See [the installation guide](https://huggingface.co/docs/lerobot/installation))

Next, we will declare some environment variables for convenience. Adjust the camera indices and robot port to match your system configuration.

```bash
export ROBOT_PORT=/dev/ttyACM0
export TELEOP_PORT=/dev/ttyACM1
export HF_USERNAME=<your_hf_username>
export ROBOT_CAMERAS="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG} }"
```

## Step 1 — Collect Data

```bash
lerobot-record \
    --robot.type=omx_follower \
    --robot.port=$ROBOT_PORT \
    --robot.id=omx_follower \
    --robot.cameras="$ROBOT_CAMERAS" \
    --teleop.type=omx_leader \
    --teleop.port=$TELEOP_PORT \
    --teleop.id=omx_leader \
    --dataset.repo_id=$HF_USERNAME/omx_pickandplace \
    --dataset.root=data/omx_pickandplace \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick the cube and place it in the blue square" \
    --dataset.streaming_encoding=true \
    --dataset.push_to_hub=true
```

### Bonus Auto-Collect script

/!\ This is specific to our setup and the task of picking and placing a cube. It is not a general-purpose data collection script. As you may notice, it doesn't require a teleop.

```bash
python -m examples.omx.record_grab \
    --robot.type=omx_follower \
    --robot.port=$ROBOT_PORT \
    --robot.id=omx_follower \
    --robot.cameras="$ROBOT_CAMERAS" \
    --dataset.repo_id=$HF_USERNAME/omx_pickandplace \
    --dataset.root=data/omx_pickandplace \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick the cube and place it in the blue square" \
    --dataset.streaming_encoding=true \
    --dataset.push_to_hub=true
```

Each episode:

1. The arm grabs the cube from the center of the workspace and places it at a random position.
2. The arm returns to HOME.
3. A targeted grab is recorded: HOME → approach raised → lower onto cube → grasp → lift → carry → drop → HOME.

A dataset is already available here [`maximellerbach/omx_pickandplace`](https://huggingface.co/datasets/maximellerbach/omx_pickandplace), so you can skip directly to training if you want.

## Step 2 — Train

To train a simple `ACT` policy on the collected dataset, you can use the `lerobot-train` CLI:

```bash
lerobot-train \
    --dataset.repo_id=$HF_USERNAME/omx_pickandplace \
    --policy.type=act \
    --output_dir=outputs/train/omx_pickandplace_act \
    --policy.device=cuda \
    --policy.repo_id=$HF_USERNAME/omx_pickandplace_act \
    --steps=20000 \
    --wandb.enable=true
```

A pretrained `ACT` policy is already available here [`maximellerbach/omx_pickandplace_act`](https://huggingface.co/maximellerbach/omx_pickandplace_act).

## Step 3 — Rollout

Use the `lerobot-rollout` CLI with base strategy:

```bash
lerobot-rollout \
    --strategy.type=base \
    --robot.type=omx_follower \
    --robot.port=$ROBOT_PORT \
    --robot.id=omx_follower \
    --robot.cameras="$ROBOT_CAMERAS" \
    --policy.path=$HF_USERNAME/omx_pickandplace_act \
```

For continuous recording with automatic upload (sentry mode):

```bash
lerobot-rollout \
    --strategy.type=sentry \
    --strategy.upload_every_n_episodes=10 \
    --robot.type=omx_follower \
    --robot.port=$ROBOT_PORT \
    --robot.id=omx_follower \
    --robot.cameras="$ROBOT_CAMERAS" \
    --policy.path=$HF_USERNAME/omx_pickandplace_act \
    --dataset.repo_id=$HF_USERNAME/rollout_omx_pickandplace_act \
```

## Environment Reset Utility

Those are specific to this particular physical setup. Those are scripts that execute hardcoded sequences of actions on the robot to reset the environment, which is useful for data collection and evaluation. They are not general-purpose scripts.

`reset_environment.py` can be run standalone to prepare the workspace:

```bash
# Grab cube + place it at a random position on the left side
python -m examples.omx.reset_environment --port $ROBOT_PORT --mode grab_and_place
```

It also exposes `grab_cube(robot)` and `place_cube(robot)` for use in custom scripts.
