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
{% elif model_name == "gaussian_actor" %}
This is a Gaussian Actor policy (Gaussian policy with a tanh squash) — the policy-side component used by [Soft Actor-Critic (SAC)](https://huggingface.co/papers/1801.01290) and related maximum-entropy continuous-control algorithms.
{% elif model_name == "pi0_fast" %}
[π₀-FAST (Pi0-FAST)](https://www.physicalintelligence.company/research/fast) is a Vision-Language-Action model for general robot control, from Physical Intelligence. It models continuous robot actions with autoregressive next-token prediction using FAST (Frequency-space Action Sequence Tokenization), training up to 5x faster than diffusion-based π₀.
{% elif model_name == "eo1" %}
[EO-1](https://huggingface.co/papers/2508.21112) is a Vision-Language-Action model for general robot control. It pairs a Qwen2.5-VL backbone for vision-language understanding with a continuous flow-matching action head that denoises action chunks.
{% elif model_name == "groot" %}
[GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T) is an open, cross-embodiment foundation model from NVIDIA for generalized humanoid robot reasoning and skills. It takes language and images as input and uses a flow-matching action transformer to predict actions conditioned on vision, language, and proprioception.
{% elif model_name == "multi_task_dit" %}
[Multi-Task Diffusion Transformer (DiT)](https://huggingface.co/papers/2507.05331) extends Diffusion Policy with a large Diffusion Transformer and text + vision conditioning for multi-task robot learning. It supports both diffusion and flow-matching objectives and reaches high dexterity with only ~450M parameters.
{% elif model_name == "wall_x" %}
[WALL-OSS](https://huggingface.co/papers/2509.11766) is an open-source foundation model for embodied intelligence from XSquare Robot. Built on Qwen2.5-VL, it uses a tightly-coupled multimodal architecture with flow matching to unify semantic reasoning and high-frequency action generation for cross-embodiment control.
{% elif model_name == "xvla" %}
[X-VLA](https://huggingface.co/papers/2510.10274) is a soft-prompted, flow-matching Vision-Language-Action framework that treats each robot or hardware setup as a "task" encoded with a small set of learnable Soft Prompt embeddings, letting a single model reconcile diverse robot morphologies, sensors, and action spaces.
{% else %}
This is a **{{ model_name }}** policy trained with [LeRobot](https://github.com/huggingface/lerobot).
{% endif %}

This policy has been trained and pushed to the Hub using [LeRobot](https://github.com/huggingface/lerobot).
{% set policy_docs = {"act": "act", "smolvla": "smolvla", "pi0": "pi0", "pi0_fast": "pi0fast", "pi05": "pi05", "eo1": "eo1", "groot": "groot"} %}
{% if policy_docs.get(model_name) %}Learn how to train and run it in the [LeRobot {{ model_name }} guide](https://huggingface.co/docs/lerobot/main/en/{{ policy_docs[model_name] }}), or browse the [full documentation](https://huggingface.co/docs/lerobot/index).
{% else %}See the [full LeRobot documentation](https://huggingface.co/docs/lerobot/index).
{% endif %}

---

## Model Details

- **License:** {{ license | default("\[More Information Needed]", true) }}
{% if robot_type %}- **Robot type:** `{{ robot_type }}`
{% endif %}{% if cameras %}- **Cameras:** {% for camera in cameras %}`{{ camera }}`{% if not loop.last %}, {% endif %}{% endfor %}
{% endif %}
{% if input_features or output_features %}
## Inputs & Outputs

The policy consumes these observation features and produces these action features.
{% if input_features %}
**Inputs**

| Feature | Type | Shape |
| --- | --- | --- |
{% for name, feature in input_features.items() %}| `{{ name }}` | {{ feature.type.value }} | `{{ feature.shape }}` |
{% endfor %}{% endif %}{% if output_features %}
**Outputs**

| Feature | Type | Shape |
| --- | --- | --- |
{% for name, feature in output_features.items() %}| `{{ name }}` | {{ feature.type.value }} | `{{ feature.shape }}` |
{% endfor %}{% endif %}{% endif %}
{% if dataset %}
## Training Dataset

- **Repository:** [{{ dataset.repo_id }}](https://huggingface.co/datasets/{{ dataset.repo_id }})
- **Episodes:** {{ dataset.episodes }}
- **Frames:** {{ dataset.frames }}
- **Frame rate:** {{ dataset.fps }} FPS
{% if dataset.tasks %}- **Task(s):** {% for task in dataset.tasks %}"{{ task }}"{% if not loop.last %}, {% endif %}{% endfor %}
{% endif %}{% endif %}
{% if training %}
## Training Configuration

| Setting | Value |
| --- | --- |
| Training steps | {{ training.steps }} |
| Batch size | {{ training.batch_size }} |
{% if training.optimizer %}| Optimizer | {{ training.optimizer }} |
{% endif %}{% if training.lr %}| Learning rate | {{ training.lr }} |
{% endif %}{% if training.seed is not none %}| Seed | {{ training.seed }} |
{% endif %}| LeRobot version | {{ training.lerobot_version }} |
{% endif %}
---

## How to Get Started with the Model

New to LeRobot? These guides cover the full workflow:

- **[Install LeRobot](https://huggingface.co/docs/lerobot/main/en/installation)** — set up the `lerobot` package.
- **[Hardware setup](https://huggingface.co/docs/lerobot/main/en/hardware_guide)** — assemble, wire, and calibrate your robot and cameras.
- **[Record data & train a policy](https://huggingface.co/docs/lerobot/en/il_robots)** — the end-to-end imitation-learning walkthrough.
- **[CLI cheat-sheet](https://huggingface.co/docs/lerobot/main/en/cheat-sheet)** — quick reference for the `lerobot-*` commands.

The short version to train and run this policy:

### Train from scratch

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/<dataset> \
  --policy.type={{ model_name }} \
  --output_dir=outputs/train/<desired_policy_repo_id> \
  --job_name=lerobot_training \
  --policy.device=cuda \
  --policy.repo_id=${HF_USER}/<desired_policy_repo_id> \
  --wandb.enable=true
```

_Writes checkpoints to `outputs/train/<desired_policy_repo_id>/checkpoints/`._

### Evaluate the policy/run inference

```bash
lerobot-rollout \
  --strategy.type=base \
  --robot.type=<your_robot_type> \
  --robot.port=<your_robot_port> \
  --robot.cameras="{ <camera_1>: {type: opencv, index_or_path: <index_or_path>, width: 640, height: 480, fps: 30}, <camera_2>: {type: opencv, index_or_path: <index_or_path>, width: 640, height: 480, fps: 30}}" \
  --policy.path=<hf_user>/<desired_policy_repo_id> \
  --task="<your_task_description>" \
  --duration=60
```

Replace every `<...>` placeholder with your own values. The `--robot.type`, `--robot.port`, and camera names/indices must match the robot and observation keys this policy was trained on, and `--task` should describe what you want the policy to do.

When `--strategy.type=base` is used the script doesn't record the episodes. Skipping duration will make the policy run indefinitely. For more information look at [rollout documentation](https://huggingface.co/docs/lerobot/main/en/inference).

---

## Evaluation

<!-- Add evaluation results here: success rate, number of trials, and the conditions (robot, task, environment). -->

_No evaluation results have been provided for this policy yet._

---

## Citation

If you use this policy, please cite the method linked in the description above, along with LeRobot:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```
