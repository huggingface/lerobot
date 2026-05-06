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
{% else %}
_Reward model type not recognized — please update this template._
{% endif %}

This reward model has been trained and pushed to the Hub using [LeRobot](https://github.com/huggingface/lerobot).
See the full documentation at [LeRobot Docs](https://huggingface.co/docs/lerobot/index).

---

## How to Get Started with the Reward Model

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

### Load the reward model in Python

```python
from lerobot.rewards import make_reward_model

reward_model = make_reward_model(pretrained_path="<hf_user>/<reward_model_repo_id>")
reward = reward_model.compute_reward(batch)
```

---

## Model Details

- **License:** {{ license | default("\[More Information Needed]", true) }}
