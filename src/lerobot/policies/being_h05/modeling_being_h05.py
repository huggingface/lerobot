from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from collections import deque
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import require_package

from .configuration_being_h05 import BeingH05Config


def _prepare_author_imports() -> None:
    """Bridge the audited source across supported Transformers/FlashAttention installs."""
    # Populate the optional-dependency cache before installing the import-only stub.
    import transformers.activations  # noqa: F401
    import transformers.utils
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if not hasattr(transformers.utils, "torch_compilable_check"):
        transformers.utils.torch_compilable_check = lambda condition, message: (
            None if condition else (_ for _ in ()).throw(ValueError(message))
        )

    # Restore the Transformers 4.57 default removed in Transformers 5.
    if "default" not in ROPE_INIT_FUNCTIONS:

        def compute_default_rope(config, device=None, seq_len=None):  # noqa: ARG001
            base = (
                config.rope_parameters["rope_theta"]
                if hasattr(config, "rope_parameters")
                else config.rope_theta
            )
            partial = getattr(config, "partial_rotary_factor", 1.0)
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            dim = int(head_dim * partial)
            exponent = torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim
            return 1.0 / (base**exponent), 1.0

        ROPE_INIT_FUNCTIONS["default"] = compute_default_rope

    if importlib.util.find_spec("flash_attn") is None:
        flash_attn = types.ModuleType("flash_attn")
        flash_attn.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)

        def flash_unavailable(*_args, **_kwargs):
            raise RuntimeError("FlashAttention is unavailable; Being-H0.5 must use eager attention.")

        flash_attn.flash_attn_varlen_func = flash_unavailable
        sys.modules["flash_attn"] = flash_attn


class BeingH05Policy(PreTrainedPolicy):
    """LeRobot owner for the unmodified Being-H0.5 architecture.

    The optional author source is imported lazily. Published checkpoints contain the
    original model below ``model.*`` and therefore round-trip through HubMixin without
    changing tensor names or shapes.
    """

    config_class = BeingH05Config
    name = "being_h05"

    def __init__(
        self,
        config: BeingH05Config,
        tokenizer_path: str | None = None,
        tokenizer_load_revision: str | None = None,
        **kwargs: Any,
    ):
        require_package("transformers", extra="being_h05")
        from transformers import AutoTokenizer, Qwen2Config, Qwen3Config
        from transformers.models.internvl.configuration_internvl import InternVLVisionConfig
        from transformers.models.internvl.modeling_internvl import InternVLVisionModel

        super().__init__(config)
        config.validate_features()
        self.config = config
        _prepare_author_imports()
        try:
            from BeingH.model import Qwen2ForCausalLM, Qwen3ForCausalLM
            from BeingH.model.beingvla import BeingH, BeingHConfig
            from BeingH.model.layers import InternVLConnector
            from BeingH.model.vit_model.internvit_navit import has_flash_attn
        except ImportError as error:
            raise ImportError(
                "Being-H0.5's audited author architecture is required. Clone "
                "https://github.com/BeingBeyond/Being-H at "
                f"{config.author_source_revision} and add its Being-H05 directory to PYTHONPATH."
            ) from error
        if not config.author_config:
            raise ValueError(
                "author_config is required; load a published LeRobot checkpoint or provide "
                "the released Being-H0.5 config payload."
            )
        author_config = BeingHConfig(**config.author_config)
        llm_dict = config.author_config["llm_config"]
        qwen3 = "Qwen3" in llm_dict.get("layer_module", "")
        llm_config_cls = Qwen3Config if qwen3 else Qwen2Config
        llm_model_cls = Qwen3ForCausalLM if qwen3 else Qwen2ForCausalLM
        llm_config = llm_config_cls.from_dict(llm_dict)
        if llm_dict.get("expert_config"):
            llm_config.expert_config = llm_config_cls.from_dict(llm_dict["expert_config"])
        vit_dict = dict(config.author_config["vit_config"])
        vit_config = InternVLVisionConfig.from_dict(vit_dict)
        vit_config.attention_bias, vit_config.use_qk_norm = (
            vit_dict["qkv_bias"],
            vit_dict["qk_normalization"],
        )
        vit_config.hidden_dropout_prob = vit_config.projection_dropout = vit_dict["dropout"]
        vit_config.layer_scale_init_value = vit_dict["initializer_factor"]
        # Select BeingH's fallback before constructing its language child model.
        attention_implementation = "flash_attention_2" if has_flash_attn else "eager"
        llm_config._attn_implementation = attention_implementation
        llm_config.attn_implementation = attention_implementation
        vit_config._attn_implementation = attention_implementation
        vit_config.use_flash_attn = has_flash_attn
        author_config.llm_config = llm_config
        author_config.vit_config = vit_config
        author_config.num_inference_timesteps = config.num_inference_steps
        language_model = llm_model_cls(llm_config)
        vit_model = InternVLVisionModel(vit_config)
        connector = InternVLConnector(
            llm_hidden_size=llm_config.hidden_size,
            vit_hidden_size=vit_config.hidden_size,
            downsample_ratio=author_config.downsample_ratio,
        )
        self.model = BeingH(
            language_model, vit_model, connector, author_config, use_flash_attn=has_flash_attn
        )
        patch_size = vit_config.patch_size[0]
        self.model.num_image_token = int(
            (author_config.force_image_size // patch_size) ** 2 * author_config.downsample_ratio**2
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or config.tokenizer_name,
            revision=tokenizer_load_revision if tokenizer_path else config.tokenizer_revision,
            use_fast=False,
            trust_remote_code=True,
        )
        special = self.tokenizer.convert_tokens_to_ids(
            ["<|im_start|>", "<|im_end|>", "<img>", "</img>", "<|state_start|>", "<|state_end|>"]
        )
        self._bos, self._eos, self._image_start, self._image_end, self._state_start, self._state_end = special
        newline = self.tokenizer.encode("\n")
        if len(newline) != 1:
            raise ValueError("Being-H0.5 checkpoint tokenizer must encode newline as one token.")
        self._newline = newline[0]
        self.reset()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, revision=None, **kwargs):
        kwargs.setdefault("tokenizer_path", str(pretrained_name_or_path))
        kwargs.setdefault("tokenizer_load_revision", revision)
        return super().from_pretrained(pretrained_name_or_path, revision=revision, **kwargs)

    def _save_pretrained(self, save_directory, state_dict: dict[str, torch.Tensor] | None = None) -> None:
        super()._save_pretrained(save_directory, state_dict=state_dict)
        self.tokenizer.save_pretrained(save_directory)

    def reset(self) -> None:
        self._action_queue: deque[torch.Tensor] = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        return self.parameters()

    def _author_kwargs(self, batch: dict[str, Any]) -> dict[str, Any]:
        kwargs = batch.get("being_h05.author_inputs")
        return kwargs if kwargs is not None else self._pack_author_inputs(batch, training=ACTION in batch)

    def _pack_author_inputs(self, batch: dict[str, Any], training: bool) -> dict[str, Any]:
        states = batch["being_h05.state"]
        pixels = batch["being_h05.pixel_values"]
        prompts = batch["being_h05_prompt"]
        device = states.device
        bsz, views = pixels.shape[:2]
        image_valid = batch.get(
            "being_h05.image_valid",
            torch.ones((bsz, views), dtype=torch.bool, device=device),
        )
        text_ids: list[int] = []
        text_indexes: list[int] = []
        vision_indexes: list[int] = []
        state_indexes: list[int] = []
        action_indexes: list[int] = []
        position_ids: list[int] = []
        sample_lens: list[int] = []
        split_lens: list[int] = []
        attn_modes: list[str] = []
        packed_images: list[torch.Tensor] = []
        cursor = 0
        system_ids = self.tokenizer.encode(f"system\n{self.model.system_message}")
        user_ids = self.tokenizer.encode("user\n")
        assistant_ids = self.tokenizer.encode("assistant\n")
        for sample in range(bsz):
            sample_images = pixels[sample, image_valid[sample]]
            if sample_images.shape[0] == 0:
                raise ValueError("Being-H0.5 requires at least one present camera per sample.")
            packed_images.extend(sample_images.unbind(0))
            num_image_tokens = self.model.num_image_token * sample_images.shape[0]
            sample_start = cursor
            rope = 0
            block = [self._bos, *system_ids, self._eos, self._newline]
            text_ids.extend(block)
            text_indexes.extend(range(cursor, cursor + len(block)))
            position_ids.extend(range(rope, rope + len(block)))
            cursor += len(block)
            rope += len(block)
            split_lens.append(len(block))
            attn_modes.append("causal")

            block_start = cursor
            block = [self._bos, *user_ids, self._image_start]
            text_ids.extend(block)
            text_indexes.extend(range(cursor, cursor + len(block)))
            cursor += len(block)
            vision_indexes.extend(range(cursor, cursor + num_image_tokens))
            cursor += num_image_tokens
            text_ids.extend([self._image_end, self._state_start])
            text_indexes.extend([cursor, cursor + 1])
            cursor += 2
            state_indexes.append(cursor)
            cursor += 1
            instruction = self.tokenizer.encode(prompts[sample])
            tail = [self._state_end, *instruction, self._eos, self._newline]
            text_ids.extend(tail)
            text_indexes.extend(range(cursor, cursor + len(tail)))
            cursor += len(tail)
            content_len = cursor - block_start
            position_ids.extend(range(rope, rope + content_len))
            rope += content_len
            split_lens.append(content_len)
            attn_modes.append("causal")

            block_start = cursor
            block = [self._bos, *assistant_ids]
            text_ids.extend(block)
            text_indexes.extend(range(cursor, cursor + len(block)))
            cursor += len(block)
            action_indexes.extend(range(cursor, cursor + self.config.chunk_size))
            cursor += self.config.chunk_size
            text_ids.append(self._eos)
            text_indexes.append(cursor)
            cursor += 1
            action_len = cursor - block_start
            position_ids.extend(range(rope, rope + action_len))
            split_lens.append(action_len)
            attn_modes.append("causal")
            sample_lens.append(cursor - sample_start)

        padding = (-cursor) % 128
        if padding:
            sample_lens.append(padding)
            split_lens.append(padding)
            attn_modes.append("causal")
        result = {
            "sequence_length": cursor,
            "packed_text_ids": torch.tensor(text_ids, dtype=torch.long, device=device),
            "packed_text_indexes": torch.tensor(text_indexes, dtype=torch.long, device=device),
            "sample_lens": sample_lens,
            "packed_position_ids": torch.tensor(position_ids, dtype=torch.long, device=device),
            "split_lens": split_lens,
            "attn_modes": attn_modes,
            "packed_vit_tokens": torch.stack(packed_images).to(device),
            "packed_vit_token_indexes": torch.tensor(vision_indexes, dtype=torch.long, device=device),
            "packed_action_indexes": torch.tensor(action_indexes, dtype=torch.long, device=device),
            "padded_state": states,
            "packed_state_indexes": torch.tensor(state_indexes, dtype=torch.long, device=device),
            "embodiment_ids": torch.full((bsz,), self.config.embodiment_id, dtype=torch.long, device=device),
        }
        if training:
            actions = batch[ACTION]
            result["padded_action"] = actions.reshape(-1, actions.shape[-1])
            valid = batch.get("being_h05.action_valid", torch.ones_like(actions, dtype=torch.bool))
            result["padded_action_mask"] = valid.reshape(-1, valid.shape[-1])
        return result

    def forward(self, batch: dict[str, Any], reduction: str = "mean"):
        output = self.model(**self._author_kwargs(batch))
        if isinstance(output, dict):
            loss = output.get("loss")
            if loss is None:
                loss = output["action_loss"] + output["und_loss"]
        else:
            loss = output.loss
        if reduction == "none" and loss.ndim == 0:
            loss = loss.unsqueeze(0)
        return loss, {"loss": float(loss.detach().mean())}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], **kwargs) -> torch.Tensor:
        if batch["being_h05.state"].shape[0] == 1:
            output = self.model.get_action(**self._author_kwargs(batch), **kwargs)
            chunks = output["action_pred"].reshape(1, self.config.chunk_size, self.config.unified_action_dim)
        else:
            # The audited author generate() hard-codes B=1. Preserve its exact solver by
            # evaluating batch elements independently instead of changing its numerics.
            chunk_list = []
            for index in range(batch["being_h05.state"].shape[0]):
                single = {
                    key: (value[index : index + 1] if isinstance(value, torch.Tensor) else [value[index]])
                    for key, value in batch.items()
                }
                output = self.model.get_action(**self._author_kwargs(single), **kwargs)
                chunk_list.append(
                    output["action_pred"].reshape(1, self.config.chunk_size, self.config.unified_action_dim)
                )
            chunks = torch.cat(chunk_list)
        return chunks

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs) -> torch.Tensor:
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()
