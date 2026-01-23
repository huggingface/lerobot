---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
# prettier-ignore
{{card_data}}
---

# Model Card for {{ model_name | default("Model ID", true) }}

<!-- Provide a quick summary of what the model is/does. -->

{% if model_name == "smolvla" %}
[SmolVLA](https://huggingface.co/papers/2506.01844) is a compact, efficient vision-language-action model that achieves competitive performance at reduced computational costs and can be deployed on consumer-grade hardware.
{% elif model_name == "act" %}
[Action Chunking with Transformers (ACT)](https://huggingface.co/papers/2304.13705) is an imitation-learning method that predicts short action chunks instead of single steps. It learns from teleoperated data and often achieves high success rates.
{% elif model_name == "tdmpc" %}
[TD-MPC](https://huggingface.co/papers/2203.04955) combines model-free and model-based approaches to improve sample efficiency and performance in continuous control tasks by using a learned latent dynamics model and terminal value function.
{% elif model_name == "diffusion" %}
[Diffusion Policy](https://huggingface.co/papers/2303.04137) treats visuomotor control as a generative diffusion process, producing smooth, multi-step action trajectories that excel at contact-rich manipulation.
{% elif model_name == "vqbet" %}
[VQ-BET](https://huggingface.co/papers/2403.03181) combines vector-quantised action tokens with Behaviour Transformers to discretise control and achieve data-efficient imitation across diverse skills.
{% elif model_name == "pi0" %}
**π₀ (Pi0)**

π₀ is a Vision-Language-Action model for general robot control, from Physical Intelligence. The LeRobot implementation is adapted from their open source OpenPI repository.

**Model Overview**

π₀ represents a breakthrough in robotics as the first general-purpose robot foundation model developed by Physical Intelligence. Unlike traditional robots that are narrow specialists programmed for repetitive motions, π₀ is designed to be a generalist policy that can understand visual inputs, interpret natural language instructions, and control a variety of different robots across diverse tasks.

For more details, see the [Physical Intelligence π₀ blog post](https://www.physicalintelligence.company/blog/pi0).
{% elif model_name == "pi05" %}
**π₀.₅ (Pi05) Policy**

π₀.₅ is a Vision-Language-Action model with open-world generalization, from Physical Intelligence. The LeRobot implementation is adapted from their open source OpenPI repository.

**Model Overview**

π₀.₅ represents a significant evolution from π₀, developed by Physical Intelligence to address a big challenge in robotics: open-world generalization. While robots can perform impressive tasks in controlled environments, π₀.₅ is designed to generalize to entirely new environments and situations that were never seen during training.

For more details, see the [Physical Intelligence π₀.₅ blog post](https://www.physicalintelligence.company/blog/pi05).
{% elif model_name == "sac" %}
[Soft Actor-Critic (SAC)](https://huggingface.co/papers/1801.01290) is an entropy-regularised actor-critic algorithm offering stable, sample-efficient learning in continuous-control environments.
{% elif model_name == "reward_classifier" %}
A reward classifier is a lightweight neural network that scores observations or trajectories for task success, providing a learned reward signal or offline evaluation when explicit rewards are unavailable.
{% else %}
_Model type not recognized — please update this template._
{% endif %}

This policy has been trained and pushed to the Hub using [LeRobot](https://github.com/huggingface/lerobot).
See the full documentation at [LeRobot Docs](https://huggingface.co/docs/lerobot/index).

---

## How to Get Started with the Model

For a complete walkthrough, see the [training guide](https://huggingface.co/docs/lerobot/il_robots#train-a-policy).
Below is the short version on how to train and run inference/eval:

### Train from scratch

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/<dataset> \
  --policy.type=act \
  --output_dir=outputs/train/<desired_policy_repo_id> \
  --job_name=lerobot_training \
  --policy.device=cuda \
  --policy.repo_id=${HF_USER}/<desired_policy_repo_id>
  --wandb.enable=true
```

_Writes checkpoints to `outputs/train/<desired_policy_repo_id>/checkpoints/`._

### Evaluate the policy/run inference

```bash
lerobot-record \
  --robot.type=so100_follower \
  --dataset.repo_id=<hf_user>/eval_<dataset> \
  --policy.path=<hf_user>/<desired_policy_repo_id> \
  --episodes=10
```

Prefix the dataset repo with **eval\_** and supply `--policy.path` pointing to a local or hub checkpoint.

---

## Model Details

- **License:** {{ license | default("\[More Information Needed]", true) }}
