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
{% elif model_name == "diffusion" %}
[Diffusion Policy](https://huggingface.co/papers/2303.04137) treats visuomotor control as a generative diffusion process, producing smooth, multi-step action trajectories that excel at contact-rich manipulation.
{% elif model_name == "pi0" %}
[π₀ (Pi0)](https://www.physicalintelligence.company/blog/pi0) is a general-purpose robot foundation model from Physical Intelligence: a generalist Vision-Language-Action policy that understands visual inputs, interprets natural language instructions, and controls a variety of different robots across diverse tasks. The LeRobot implementation is adapted from their open-source OpenPI repository.
{% elif model_name == "pi05" %}
[π₀.₅ (Pi05)](https://www.physicalintelligence.company/blog/pi05) is a Vision-Language-Action model from Physical Intelligence designed for open-world generalization: it evolves π₀ to generalize to entirely new environments and situations that were never seen during training. The LeRobot implementation is adapted from their open-source OpenPI repository.
{% elif model_name == "molmoact2" %}
[MolmoAct2](https://allenai.org/blog/molmoact2) is an open robotics foundation model from the Allen Institute for AI (Ai2) that maps camera images and language instructions to robot action chunks. The LeRobot implementation supports training and evaluation of the regular MolmoAct2 model.
{% elif model_name == "vla_jepa" %}
[VLA-JEPA](https://arxiv.org/abs/2602.10098) is a Vision-Language-Action model that combines a Qwen3-VL language backbone with a self-supervised video world model (V-JEPA2) and a flow-matching DiT action head.
{% elif model_name == "gaussian_actor" %}
This is a Gaussian Actor policy (Gaussian policy with a tanh squash) — the policy-side component used by [Soft Actor-Critic (SAC)](https://huggingface.co/papers/1801.01290) and related maximum-entropy continuous-control algorithms.
{% elif model_name == "pi0_fast" %}
[π₀-FAST (Pi0-FAST)](https://www.physicalintelligence.company/research/fast) is a Vision-Language-Action model for general robot control, from Physical Intelligence. It models continuous robot actions with autoregressive next-token prediction using FAST (Frequency-space Action Sequence Tokenization), training up to 5x faster than diffusion-based π₀.
{% elif model_name == "eo1" %}
[EO-1](https://huggingface.co/papers/2508.21112) is a Vision-Language-Action model for general robot control. It pairs a Qwen2.5-VL backbone for vision-language understanding with a continuous flow-matching action head that denoises action chunks.
{% elif model_name == "groot" %}
[GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) is an open, cross-embodiment foundation model from NVIDIA for generalized humanoid robot reasoning and skills. It uses a Cosmos-Reason2/Qwen3-VL backbone and a flow-matching action transformer to predict actions conditioned on vision, language, and proprioception.
{% elif model_name == "multi_task_dit" %}
[Multi-Task Diffusion Transformer (DiT)](https://huggingface.co/papers/2507.05331) extends Diffusion Policy with a large Diffusion Transformer and text + vision conditioning for multi-task robot learning. It supports both diffusion and flow-matching objectives and reaches high dexterity with only ~450M parameters.
{% elif model_name == "wall_x" %}
[WALL-OSS](https://huggingface.co/papers/2509.11766) is an open-source foundation model for embodied intelligence from XSquare Robot. Built on Qwen2.5-VL, it uses a tightly-coupled multimodal architecture with flow matching to unify semantic reasoning and high-frequency action generation for cross-embodiment control.
{% elif model_name == "xvla" %}
[X-VLA](https://huggingface.co/papers/2510.10274) is a soft-prompted, flow-matching Vision-Language-Action framework that treats each robot or hardware setup as a "task" encoded with a small set of learnable Soft Prompt embeddings, letting a single model reconcile diverse robot morphologies, sensors, and action spaces.
{% elif model_name == "evo1" %}
[EVO1](https://github.com/MINT-SJTU/Evo-1) is a Vision-Language-Action policy built around an InternVL3 backbone and a continuous flow-matching action head. It embeds camera images and the language instruction with InternVL3 and predicts future action chunks via flow matching.
{% elif model_name == "fastwam" %}
[FastWAM](https://arxiv.org/abs/2603.16666) is a World Action Model policy that keeps video world-modeling during training but predicts actions directly at inference time, initializing its visual world-model components from the Wan2.2 video-diffusion stack.
{% elif model_name == "lingbot_va" %}
[LingBot-VA](https://github.com/Robbyant/lingbot-va) is an autoregressive video-action world-model policy built on the Wan2.2 video-diffusion stack. It interleaves the prediction of future video latents and robot actions in a single autoregressive sequence, feeding observed keyframes back into its KV cache for closed-loop world modeling.
{% else %}
This is a **{{ model_name }}** policy trained with [LeRobot](https://github.com/huggingface/lerobot).
{% endif %}
{% set diagrams = {
  "smolvla": "https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/aooU0a3DMtYmy_1IWMaIM.png",
  "pi0": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/lerobot-pi0%20(1).png",
  "pi0_fast": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/lerobot-pifast.png",
  "eo1": "https://huggingface.co/datasets/HaomingSong/lerobot-documentation-images/resolve/main/lerobot/eo_pipeline.png",
  "groot": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/lerobot-groot-paper1%20(1).png",
  "wall_x": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/walloss-lerobot-paper.png",
  "xvla": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/xvla-architecture.png"
} %}
{% if diagrams.get(model_name) %}
<p align="center">
  <img src="{{ diagrams[model_name] }}" alt="{{ model_name }} architecture" width="85%"/>
</p>
{% endif %}

<!-- A short demo is worth more than any description! Record a GIF/video of the policy
running on your robot, upload it to this repo, and embed it here:
<p align="center">
  <img src="https://huggingface.co/<hf_user>/<policy_repo_id>/resolve/main/demo.gif" width="60%"/>
</p>
-->

This policy has been trained and pushed to the Hub using [LeRobot](https://github.com/huggingface/lerobot).
{% set policy_docs = {
  "act": "act",
  "smolvla": "smolvla",
  "pi0": "pi0",
  "pi0_fast": "pi0fast",
  "pi05": "pi05",
  "molmoact2": "molmoact2",
  "vla_jepa": "vla_jepa",
  "eo1": "eo1",
  "groot": "groot",
  "xvla": "xvla",
  "multi_task_dit": "multi_task_dit",
  "wall_x": "walloss",
  "evo1": "evo1",
  "fastwam": "fastwam",
  "lingbot_va": "lingbot_va"
} %}
{% if policy_docs.get(model_name) %}Learn how to train and run it in the [LeRobot {{ model_name }} guide](https://huggingface.co/docs/lerobot/main/en/{{ policy_docs[model_name] }}), or browse the [full documentation](https://huggingface.co/docs/lerobot/index).
{% else %}See the [full LeRobot documentation](https://huggingface.co/docs/lerobot/index).
{% endif %}

---

## Model Details

- **License:** {{ license | default("\[More Information Needed]", true) }}
{% if base_model %}- **Fine-tuned from:** [{{ base_model }}](https://huggingface.co/{{ base_model }})
{% endif %}{% if robot_type %}- **Robot type:** `{{ robot_type }}`
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
{% endif %}
<a class="flex" href="https://huggingface.co/spaces/lerobot/visualize_dataset?path={{ dataset.repo_id }}">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl.svg"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl-dark.svg"/>
</a>
{% endif %}
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

The short version to run and train this policy:

### Run the policy on your robot

```bash
lerobot-rollout \
  --strategy.type=base \
  --robot.type={{ robot_type | default("<your_robot_type>", true) }} \
  --robot.port=<your_robot_port> \
  --robot.cameras="{ <camera_1>: {type: opencv, index_or_path: <index_or_path>, width: 640, height: 480, fps: 30}, <camera_2>: {type: opencv, index_or_path: <index_or_path>, width: 640, height: 480, fps: 30}}" \
  --policy.path={{ policy_repo_id | default("<hf_user>/<policy_repo_id>", true) }} \
  --task="{% if dataset and dataset.tasks %}{{ dataset.tasks[0] }}{% else %}<your_task_description>{% endif %}" \
  --duration=60
```

Replace the remaining `<...>` placeholders with your own values: `--robot.port` and the camera names/indices are specific to your machine, and the camera names must match the observation keys this policy was trained on.

When `--strategy.type=base` is used the script doesn't record the episodes. Skipping duration will make the policy run indefinitely. For more information look at [rollout documentation](https://huggingface.co/docs/lerobot/main/en/inference).

{% if base_model %}### Train your own policy

This policy type is usually fine-tuned from the pretrained base model [{{ base_model }}](https://huggingface.co/{{ base_model }}):

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/<dataset> \
  --policy.path={{ base_model }} \
  --output_dir=outputs/train/<policy_repo_id> \
  --job_name=lerobot_training \
  --policy.device=cuda \
  --policy.repo_id=${HF_USER}/<policy_repo_id> \
  --wandb.enable=true
```
{% else %}### Train your own policy

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/<dataset> \
  --policy.type={{ model_name }} \
  --output_dir=outputs/train/<policy_repo_id> \
  --job_name=lerobot_training \
  --policy.device=cuda \
  --policy.repo_id=${HF_USER}/<policy_repo_id> \
  --wandb.enable=true
```
{% endif %}
_Writes checkpoints to `outputs/train/<policy_repo_id>/checkpoints/`._

---

## Evaluation

<!-- Report real-robot results here: run the policy several times per task and count the
successes. Delete the "No evaluation results" line and fill in this table instead:

| Task | Trials | Successes | Success rate |
| ---- | ------ | --------- | ------------ |
| pick the lego brick | 10 | 8 | 80% |

Also worth noting: anything that affects difficulty (new object positions, lighting,
distractors, a different robot of the same type, ...).
-->

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
