# LingBot-VLA

LingBot-VLA is an open-source vision-language-action model integrated into LeRobot as
the `lingbot_vla` policy. It pairs a **Qwen2.5-VL** multimodal backbone with a **Qwen2
action expert** and **Flow-Matching** continuous action generation over a unified 75-D
state/action space, supporting cross-embodiment and bimanual manipulation.

Full documentation: [`docs/source/lingbot_vla.mdx`](../../../../docs/source/lingbot_vla.mdx)
and the [Deployment Protocol](../../../../docs/source/lingbot_vla_deployment.mdx).

## Install

```bash
pip install -e ".[lingbot]"
```

## Usage

```bash
lerobot-train --policy.type=lingbot_vla ...
```

The vendored Qwen2.5-VL backbone lives in [`qwen_model/`](./qwen_model) and is adapted
from HuggingFace Transformers. Flash-attention is optional: the policy defaults to the
`eager` attention implementation and only uses `flash_attention_2` when explicitly
requested via the config (and when the `flash-attn` package is installed).
