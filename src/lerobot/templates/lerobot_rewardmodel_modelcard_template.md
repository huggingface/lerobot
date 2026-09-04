---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
# prettier-ignore
{{card_data}}
---

# Reward Model Card for {{ model_name | default("Reward Model ID", true) }}

<!-- Provide a quick summary of what the reward model is/does. -->

{% if model_name == "reward_classifier" %}
A reward classifier is a lightweight neural network that scores observations or trajectories for task success, providing a learned reward signal or offline evaluation when explicit rewards are unavailable.
{% elif model_name == "sarm" %}
A Success-Aware Reward Model (SARM) predicts a dense reward signal from observations, typically used downstream for reinforcement learning or human-in-the-loop fine-tuning when task success is not directly observable.
{% elif model_name == "robometer" %}
ROBOMETER is a general-purpose video-language robotic reward model built on a fine-tuned Qwen3-VL-4B backbone with progress, preference, and success heads. Given a trajectory video and a task description, it predicts dense, frame-level task progress in [0, 1] and frame-level success probabilities for downstream robot learning, including offline RL, online RL, data filtering and retrieval, and automated failure detection.
{% elif model_name == "topreward" %}
TOPReward is a **zero-shot** reward model that extracts token log-probabilities from an off-the-shelf vision-language model (default Qwen3-VL) as a reward signal. Given a video trajectory and a task instruction, it returns the VLM's log-likelihood of the instruction being true, with no fine-tuning required.
{% else %}
_Reward model type not recognized — please update this template._
{% endif %}

{% if model_name in ["robometer", "topreward"] %}
This inference-only reward-model integration can be loaded with [LeRobot](https://github.com/huggingface/lerobot). Training this model through `lerobot-train` is not currently supported.
{% else %}
This reward model has been trained and pushed to the Hub using [LeRobot](https://github.com/huggingface/lerobot).
{% endif %}
See the full documentation at [LeRobot Docs](https://huggingface.co/docs/lerobot/index).

---

## How to Get Started with the Reward Model

{% if model_name not in ["robometer", "topreward"] %}
### Train from scratch

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/<dataset> \
  --reward_model.type={{ model_name | default("reward_classifier", true) }} \
  --output_dir=outputs/train/<desired_reward_model_repo_id> \
  --job_name=lerobot_reward_training \
  --reward_model.device=cuda \
  --reward_model.repo_id=${HF_USER}/<desired_reward_model_repo_id> \
  --wandb.enable=true
```

_Writes checkpoints to `outputs/train/<desired_reward_model_repo_id>/checkpoints/`._
{% endif %}

### Load the reward model in Python

```python
from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards import make_reward_model

model_id = "<hf_user>/<reward_model_repo_id>"
config = RewardModelConfig.from_pretrained(model_id)
config.pretrained_path = model_id
reward_model = make_reward_model(config)

# `batch` is the output of this model's documented preprocessor.
{% if model_name == "reward_classifier" %}
reward = reward_model.predict_reward(batch)
{% elif model_name == "sarm" %}
prediction = reward_model.predict_progress(batch, head_mode="sparse")
progress = prediction.progress
{% elif model_name == "robometer" %}
prediction = reward_model.predict_progress(batch)
progress = prediction.progress
success_probability = prediction.success_probability
{% elif model_name == "topreward" %}
log_probability = reward_model.compute_log_probability(batch)
{% else %}
# Use the model's documented inference capability.
{% endif %}
```

---

## Model Details

- **License:** {{ license | default("\[More Information Needed]", true) }}
