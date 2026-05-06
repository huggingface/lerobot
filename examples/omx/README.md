# OMX Follower — Cube Grab Example

This is an example of what is possible to do with LeRobot on a phyisical Setup.
It is a WIP and being used internally at LeRobot and specific to our setup, but we hope it can be a useful reference for how to use LeRobot APIs and CLIs.

It includes an end-to-end example for the **OMX Follower** robot arm: pick and place a cube dataset, train a policy, and deploy it autonomously.

## Hardware

| Component | Value                                        |
| --------- | -------------------------------------------- |
| Robot     | OMX Follower (USB serial)                    |
| Cameras   | 2× OpenCV cameras (wrist + top-down)         |
| Port      | `/dev/ttyACM0` (adjust to match your system) |

## Scripts

| Script                 | Purpose                                                         |
| ---------------------- | --------------------------------------------------------------- |
| `reset_environment.py` | Standalone utility: sweep workspace, grab cube, place cube      |
| `record_grab.py`       | Automated data collection: reset → place → record grab episodes |
| `rollout.py`           | Deploy a trained policy on the robot                            |

## Setup

Make sure you have LeRobot installed in your env

## Step 1 — Collect Data

/!\ This is specific to our setup and the task of picking and placing a cube. It is not a general-purpose data collection script.

```bash
cd examples/omx

python record_grab.py \
    --robot.type=omx_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=omx_follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: MJPG}, top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG} }" \
    --dataset.repo_id=maximellerbach/omx_autocollect \
    --dataset.root=data/omx_collect \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick the cube and place it in the blue square" \
    --dataset.streaming_encoding=true \
    --dataset.push_to_hub=true
```

Each episode:

1. The arm grabs the cube from the center of the workspace and places it at a random position.
2. The arm returns to HOME.
3. A targeted grab is recorded: HOME → approach raised → lower onto cube → grasp → lift → carry → drop → HOME.

A dataset is already available here [`maximellerbach/omx_autocollect`](https://huggingface.co/datasets/maximellerbach/omx_autocollect), so you can skip directly to training if you want.

## Step 2 — Train

```bash
lerobot-train \
    --dataset.repo_id=<hf_username>/omx_pickandplace \
    --policy.type=act \
    --output_dir=outputs/train/omx_pickandplace_act \
    --wandb.enable=true
```

## Step 3 — Rollout

use the `lerobot-rollout` CLI:

```bash
lerobot-rollout \
    --strategy.type=base \
    --robot.type=omx_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=omx_follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: MJPG}, top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG} }" \
    --policy.path=<hf_username>/<model_repo_id>
```

For continuous recording with automatic upload (sentry mode):

```bash
lerobot-rollout \
    --strategy.type=sentry \
    --strategy.upload_every_n_episodes=10 \
    --robot.type=omx_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=omx_follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: MJPG}, top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG} }" \
    --policy.path=<hf_username>/<model_repo_id> \
    --dataset.repo_id=<hf_username>/rollout_omx_grab
```

## Environment Reset Utility

Those are specific to this particular setup. Those are scripts that execute hardcoded sequences of actions on the robot to reset the environment, which is useful for data collection and evaluation. They are not general-purpose scripts.

`reset_environment.py` can be run standalone to prepare the workspace:

```bash
# Sweep the workspace (cube back to center)
python reset_environment.py --port /dev/ttyACM0 --mode reset

# Grab cube + place it at a random position
python reset_environment.py --port /dev/ttyACM0 --mode grab_and_place
```

It also exposes `grab_cube(robot)` and `place_cube(robot)` for use in custom scripts.
