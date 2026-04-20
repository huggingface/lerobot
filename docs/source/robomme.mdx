# RoboMME

[RoboMME](https://robomme.github.io) is a memory-augmented manipulation benchmark built on ManiSkill (SAPIEN). It evaluates a robot's ability to retain and use information across an episode — counting, object permanence, reference, and imitation.

- **16 tasks** across 4 memory-skill suites
- **1,600 training demos** (100 per task, 50 val, 50 test)
- **Dataset**: [`lerobot/robomme`](https://huggingface.co/datasets/lerobot/robomme) — LeRobot v3.0, 768K frames at 10 fps
- **Simulator**: ManiSkill / SAPIEN, Panda arm, Linux only

![RoboMME benchmark tasks overview](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2603.04639/gradient.png)

## Tasks

| Suite                             | Tasks                                                         |
| --------------------------------- | ------------------------------------------------------------- |
| **Counting** (temporal memory)    | BinFill, PickXtimes, SwingXtimes, StopCube                    |
| **Permanence** (spatial memory)   | VideoUnmask, VideoUnmaskSwap, ButtonUnmask, ButtonUnmaskSwap  |
| **Reference** (object memory)     | PickHighlight, VideoRepick, VideoPlaceButton, VideoPlaceOrder |
| **Imitation** (procedural memory) | MoveCube, InsertPeg, PatternLock, RouteStick                  |

## Installation

> RoboMME requires **Linux** (ManiSkill/SAPIEN uses Vulkan rendering). Docker is recommended to isolate dependency conflicts.

### Native (Linux)

```bash
pip install --override <(printf 'gymnasium==0.29.1\nnumpy==1.26.4\n') \
  -e '.[smolvla,av-dep]' \
  'robomme @ git+https://github.com/RoboMME/robomme_benchmark.git@main'
```

> **Dependency note**: `mani-skill` (pulled by `robomme`) pins `gymnasium==0.29.1` and `numpy<2.0.0`, which conflict with lerobot's base `numpy>=2.0.0`. That's why `robomme` is not a pyproject extra — use the override install above, or the Docker approach below to avoid conflicts entirely.

### Docker (recommended)

```bash
# Build base image first (from repo root)
docker build -f docker/Dockerfile.eval-base -t lerobot-eval-base .

# Build RoboMME eval image (applies gymnasium + numpy pin overrides)
docker build -f docker/Dockerfile.benchmark.robomme -t lerobot-robomme .
```

The `docker/Dockerfile.benchmark.robomme` image overrides `gymnasium==0.29.1` and `numpy==1.26.4` after lerobot's install. Both versions are runtime-safe for lerobot's actual API usage.

## Running Evaluation

### Default (single task, single episode)

```bash
lerobot-eval \
    --policy.path=<your_policy_repo> \
    --env.type=robomme \
    --env.task=PickXtimes \
    --env.dataset_split=test \
    --env.task_ids=[0] \
    --eval.batch_size=1 \
    --eval.n_episodes=1
```

### Multi-task evaluation

Evaluate multiple tasks in one run by comma-separating task names. Use `task_ids` to control which episodes are evaluated per task. Recommended: 50 episodes per task for the test split.

```bash
lerobot-eval \
    --policy.path=<your_policy_repo> \
    --env.type=robomme \
    --env.task=PickXtimes,BinFill,StopCube,MoveCube,InsertPeg \
    --env.dataset_split=test \
    --env.task_ids=[0,1,2,3,4,5,6,7,8,9] \
    --eval.batch_size=1 \
    --eval.n_episodes=50
```

### Key CLI options for `env.type=robomme`

| Option               | Default       | Description                                        |
| -------------------- | ------------- | -------------------------------------------------- |
| `env.task`           | `PickXtimes`  | Any of the 16 task names above (comma-separated)   |
| `env.dataset_split`  | `test`        | `train`, `val`, or `test`                          |
| `env.action_space`   | `joint_angle` | `joint_angle` (8-D) or `ee_pose` (7-D)             |
| `env.episode_length` | `300`         | Max steps per episode                              |
| `env.task_ids`       | `null`        | List of episode indices to evaluate (null = `[0]`) |

## Dataset

The dataset [`lerobot/robomme`](https://huggingface.co/datasets/lerobot/robomme) is in **LeRobot v3.0 format** and can be loaded directly:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/robomme")
```

### Dataset features

| Feature            | Shape         | Description                     |
| ------------------ | ------------- | ------------------------------- |
| `image`            | (256, 256, 3) | Front camera RGB                |
| `wrist_image`      | (256, 256, 3) | Wrist camera RGB                |
| `actions`          | (8,)          | Joint angles + gripper          |
| `state`            | (8,)          | Joint positions + gripper state |
| `simple_subgoal`   | str           | High-level language annotation  |
| `grounded_subgoal` | str           | Grounded language annotation    |
| `episode_index`    | int           | Episode ID                      |
| `frame_index`      | int           | Frame within episode            |

### Feature key alignment (training)

The env wrapper exposes `pixels/image` and `pixels/wrist_image` as observation keys. The `features_map` in `RoboMMEEnv` maps these to `observation.images.image` and `observation.images.wrist_image` for the policy. State is exposed as `agent_pos` and maps to `observation.state`.

The dataset's `image` and `wrist_image` columns already align with the policy input keys, so no renaming is needed when fine-tuning.

## Action Spaces

| Type          | Dim | Description                                               |
| ------------- | --- | --------------------------------------------------------- |
| `joint_angle` | 8   | 7 joint angles + 1 gripper (−1 closed, +1 open, absolute) |
| `ee_pose`     | 7   | xyz + roll/pitch/yaw + gripper                            |

Set via `--env.action_space=joint_angle` (default) or `--env.action_space=ee_pose`.

## Platform Notes

- **Linux only**: ManiSkill requires SAPIEN/Vulkan. macOS and Windows are not supported.
- **GPU recommended**: Rendering is CPU-capable but slow; CUDA + Vulkan gives full speed.
- **gymnasium / numpy conflict**: See installation note above. Docker image handles this automatically.
- **ManiSkill fork**: `robomme` depends on a specific ManiSkill fork (`YinpeiDai/ManiSkill`), pulled in automatically via the `robomme` package.
