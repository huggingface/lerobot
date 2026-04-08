# RoboCerebra Benchmark

RoboCerebra is a benchmark for evaluating long-horizon robotic manipulation with vision-language models. It contains **10 tasks** spanning kitchen, living-room, and study environments, designed to require deliberative, multi-step reasoning rather than reactive policies.

- **Paper**: [RoboCerebra](https://robocerebra.github.io)
- **GitHub**: [qiuboxiang/RoboCerebra](https://github.com/qiuboxiang/RoboCerebra)
- **Dataset (LeRobot v3.0)**: [`CollisionCode/RoboCerebra_lerobot_v3.0`](https://huggingface.co/datasets/CollisionCode/RoboCerebra_lerobot_v3.0)

## Installation

RoboCerebra runs on top of the [LIBERO](https://libero-project.github.io) environment, which requires Linux.

```bash
pip install "lerobot[robocerebra]"
```

> **macOS users**: LIBERO (and therefore RoboCerebra) requires Linux. Use Docker or a remote Linux machine for evaluation.

## Dataset

The dataset is already in **LeRobot v3.0 format** — no conversion is needed.

| Property         | Value                                                                                         |
| ---------------- | --------------------------------------------------------------------------------------------- |
| HuggingFace repo | `CollisionCode/RoboCerebra_lerobot_v3.0`                                                      |
| Format           | LeRobot v3.0                                                                                  |
| Robot            | Franka Panda (7-DOF arm + gripper)                                                            |
| Action space     | 7-DOF delta end-effector pose + gripper                                                       |
| Cameras          | `observation.images.image` (agent-view), `observation.images.wrist_image` (wrist)             |
| Resolution       | 256 × 256                                                                                     |
| FPS              | 20                                                                                            |
| Task types       | Ideal, Memory_Execution, Memory_Exploration, Mix, Observation_Mismatching, Random_Disturbance |

### Loading the dataset

```python
from lerobot.datasets import LeRobotDataset

# Load the "Ideal" task type
dataset = LeRobotDataset("CollisionCode/RoboCerebra_lerobot_v3.0", root="Ideal")
```

## Available Tasks

RoboCerebra evaluates on the **libero_10** suite (10 long-horizon tasks):

| Task ID | Name                                                                                                         |
| ------- | ------------------------------------------------------------------------------------------------------------ |
| 0       | KITCHEN_SCENE3 — turn on the stove and put the moka pot on it                                                |
| 1       | KITCHEN_SCENE4 — put the black bowl in the bottom drawer of the cabinet and close it                         |
| 2       | KITCHEN_SCENE6 — put the yellow and white mug in the microwave and close it                                  |
| 3       | KITCHEN_SCENE8 — put both moka pots on the stove                                                             |
| 4       | LIVING_ROOM_SCENE1 — put both the alphabet soup and the cream cheese box in the basket                       |
| 5       | LIVING_ROOM_SCENE2 — put both the alphabet soup and the tomato sauce in the basket                           |
| 6       | LIVING_ROOM_SCENE2 — put both the cream cheese box and the butter in the basket                              |
| 7       | LIVING_ROOM_SCENE5 — put the white mug on the left plate and put the yellow and white mug on the right plate |
| 8       | LIVING_ROOM_SCENE6 — put the white mug on the plate and put the chocolate pudding to the right of the plate  |
| 9       | STUDY_SCENE1 — pick up the book and place it in the back compartment of the caddy                            |

## Running Evaluation

### Quick smoke test (1 episode)

```bash
lerobot-eval \
    --policy.path=<your_policy_on_hub> \
    --env.type=robocerebra \
    --env.task=libero_10 \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --eval.use_async_envs=false \
    --policy.device=cuda
```

### Full benchmark (all 10 tasks, parallel envs)

```bash
lerobot-eval \
    --policy.path=<your_policy_on_hub> \
    --env.type=robocerebra \
    --env.task=libero_10 \
    --eval.batch_size=10 \
    --eval.n_episodes=50 \
    --eval.use_async_envs=true \
    --policy.device=cuda
```

### Evaluating a specific subset of tasks

```bash
lerobot-eval \
    --policy.path=<your_policy_on_hub> \
    --env.type=robocerebra \
    --env.task=libero_10 \
    --env.task_ids="[0,1,2]" \
    --eval.batch_size=1 \
    --eval.n_episodes=10 \
    --policy.device=cuda
```

### Custom camera name mapping

By default, `RoboCerebraEnv` maps LIBERO camera names to match the dataset:

| LIBERO camera              | Policy key                       |
| -------------------------- | -------------------------------- |
| `agentview_image`          | `observation.images.image`       |
| `robot0_eye_in_hand_image` | `observation.images.wrist_image` |

If your policy was trained with different camera names (e.g., `camera1`/`camera2`), override with:

```bash
lerobot-eval \
    --policy.path=<your_policy_on_hub> \
    --env.type=robocerebra \
    --env.task=libero_10 \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --policy.device=cuda \
    '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
```

> **Shell quoting note**: wrap the JSON mapping in single quotes on Linux/macOS, or use `^"..."^` on Windows cmd.

## Configuration reference

All fields of `RoboCerebraEnv` can be overridden via CLI:

| Field                 | Default              | Description                                  |
| --------------------- | -------------------- | -------------------------------------------- |
| `task`                | `"libero_10"`        | LIBERO suite name                            |
| `task_ids`            | `null` (all 10)      | List of task IDs to evaluate                 |
| `fps`                 | `20`                 | Environment FPS (matches dataset)            |
| `episode_length`      | `null` (520)         | Max steps per episode                        |
| `obs_type`            | `"pixels_agent_pos"` | `"pixels"` or `"pixels_agent_pos"`           |
| `observation_height`  | `256`                | Camera height in pixels                      |
| `observation_width`   | `256`                | Camera width in pixels                       |
| `camera_name_mapping` | see above            | LIBERO cam → policy key mapping              |
| `control_mode`        | `"relative"`         | `"relative"` or `"absolute"` EEF control     |
| `init_states`         | `true`               | Use fixed initial states for reproducibility |

## Citation

If you use RoboCerebra in your work, please cite:

```bibtex
@article{robocerebra2024,
  title={RoboCerebra: A Long-Horizon Manipulation Benchmark for Evaluating Robotic Reasoning},
  author={Qiu, Boxiang and others},
  year={2024},
  url={https://robocerebra.github.io}
}
```
