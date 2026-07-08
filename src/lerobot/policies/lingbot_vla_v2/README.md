# LingBot-VLA 2.0

LingBot-VLA 2.0 is an open-source vision-language-action model integrated into LeRobot as the
`lingbot_vla_v2` policy. It pairs a **Qwen3-VL-4B** multimodal backbone with a **sparse-MoE
Qwen2 action expert** (32 experts, top-4) and **Flow-Matching** continuous action generation
over a unified 55-D state/action space, supporting cross-embodiment and bimanual manipulation.
It is the successor to the [`lingbot_vla`](../lingbot_va) (v1) policy.

Full documentation: [`docs/source/lingbot_vla_v2.mdx`](../../../../docs/source/lingbot_vla_v2.mdx).

## Install

```bash
pip install -e ".[lingbot-v2]"
```

## Usage

```bash
lerobot-train --policy.type=lingbot_vla_v2 ...
```

The vendored Qwen3-VL backbone (`qwen3vl_in_vla.py`) and sparse-MoE Qwen2 action expert
(`qwen2_action_expert.py`) are adapted from HuggingFace Transformers and the upstream
`Robbyant/lingbot-vla-v2` repository (Apache-2.0 retained). Flash-attention is optional: the
policy defaults to `eager` attention and only uses `flash_attention_2` when explicitly
requested via the config (and when the `flash-attn` package is installed).
