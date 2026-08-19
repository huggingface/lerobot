from collections.abc import Sequence

import torch
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available, require_package

if _transformers_available:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class Qwen3VLGroundingBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        torch_dtype: str = "bfloat16",
        freeze_vision_tower: bool = True,
        gradient_checkpointing: bool = True,
        lora_rank: int = 0,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_bias: str = "none",
    ):
        super().__init__()
        require_package("transformers", "cig_vla")
        dtype = getattr(torch, torch_dtype)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "right"
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.freeze_vision_tower = freeze_vision_tower
        if freeze_vision_tower:
            visual = getattr(getattr(self.model, "model", None), "visual", None)
            if visual is None:
                raise RuntimeError("Qwen3-VL visual module not found; installed transformers is incompatible")
            visual.requires_grad_(False)
        if lora_rank > 0:
            self._apply_language_lora(lora_rank, lora_alpha, lora_dropout, lora_bias)

    def _apply_language_lora(self, rank, alpha, dropout, bias):
        require_package("peft", "cig_vla")
        import re

        from peft import LoraConfig, get_peft_model

        pattern = r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
        matches = [name for name, _ in self.model.named_modules() if re.fullmatch(pattern, name)]
        if not matches:
            raise RuntimeError("No Qwen3-VL language attention modules matched LoRA target")
        if any("visual" in name for name in matches):
            raise RuntimeError("Vision module unexpectedly matched language LoRA target")
        self.lora_target_counts = {
            target: sum(name.endswith(target) for name in matches)
            for target in ("q_proj", "k_proj", "v_proj", "o_proj")
        }
        self.model = get_peft_model(
            self.model,
            LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias=bias, target_modules=pattern),
        )

    def _vision_module(self):
        model = self.model
        if hasattr(model, "base_model"):
            model = model.base_model.model
        root = getattr(model, "model", None)
        visual = getattr(root, "visual", None)
        if visual is None:
            raise RuntimeError("Qwen3-VL visual module not found")
        return visual

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_vision_tower:
            self._vision_module().eval()
        return self

    @property
    def hidden_size(self):
        return self.model.config.text_config.hidden_size

    def build_inputs(self, images: Sequence[Sequence[Tensor]], instructions: list[str]):
        messages = []
        for sample_images, instruction in zip(images, instructions, strict=True):
            content = [{"type": "image", "image": image.detach().float()} for image in sample_images]
            content.append({"type": "text", "text": instruction})
            messages.append([{"role": "user", "content": content}])
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={
                "padding": True,
                "return_tensors": "pt",
                "device": self.model.device,
                "do_rescale": False,
            },
        )
        inputs = inputs.to(self.model.device)
        self.last_input_diagnostics = {
            "token_shape": tuple(inputs["input_ids"].shape),
            "attention_shape": tuple(inputs["attention_mask"].shape),
            "pixel_values_shape": tuple(inputs["pixel_values"].shape),
            "image_grid_thw_shape": tuple(inputs["image_grid_thw"].shape),
        }
        return inputs

    def encode_multimodal(self, images, instructions):
        inputs = self.build_inputs(images, instructions)
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden = outputs.hidden_states[-1]
        self.last_hidden_shape = tuple(hidden.shape)
        return hidden, inputs["attention_mask"]

    def valid_lora_targets(self):
        candidates = ("q_proj", "k_proj", "v_proj", "o_proj")
        names = {name.rsplit(".", 1)[-1] for name, _ in self.model.named_modules()}
        targets = [name for name in candidates if name in names]
        if not targets:
            raise RuntimeError("No valid Qwen language-attention LoRA targets found")
        return targets

    def trainable_parameter_diagnostic(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total, "ratio": trainable / max(total, 1)}
