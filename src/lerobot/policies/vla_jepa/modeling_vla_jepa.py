# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoModel, AutoVideoProcessor
else:
    AutoModel = None
    AutoVideoProcessor = None

from .action_head import VLAJEPAActionHead
from .configuration_vla_jepa import VLAJEPAConfig
from .qwen_interface import Qwen3VLInterface
from .world_model import ActionConditionedVideoPredictor

# ============================================================================
# Native VLA-JEPA Model - follows original starVLA VLA_JEPA.py implementation
# ============================================================================


class VLAJEPAModel(nn.Module):
    """
    Native VLA-JEPA model following the original starVLA VLA_JEPA.py.

    Components:
      - Qwen3-VL: vision-language backbone for fused embeddings
      - DiT-B: flow-matching action head for future action prediction
      - V-JEPA: world model for video frame prediction

    Inputs are batched tensors kept on the model device
      - images: List[List[Tensor [C, H, W]]] (float [0,1]) — per sample, per view (Qwen messages)
      - instructions: List[str]
      - videos: Tensor [B, V, T, C, H, W] (float [0,1], world model only)
      - actions: Tensor [B, T, action_dim] (optional, training only)
      - state: Tensor [B, 1, state_dim] (optional)
      - action_is_pad: Tensor [B, T] (optional)
    """

    def __init__(self, config: VLAJEPAConfig) -> None:
        super().__init__()
        require_package("transformers", extra="vla_jepa")
        self.config = config

        # Vision-language backbone
        self.qwen = Qwen3VLInterface(config)

        # Tokenizer expansion for special action tokens
        self.action_tokens, self.action_token_ids, self.embodied_action_token_id = (
            self.qwen.expand_tokenizer()
        )
        self.register_buffer(
            "_action_token_ids_t",
            torch.tensor(self.action_token_ids, dtype=torch.long),
            persistent=False,
        )

        # Action head (flow-matching DiT)
        self.action_model = VLAJEPAActionHead(config, cross_attention_dim=self.qwen.model.config.hidden_size)

        # JEPA world model components
        if config.enable_world_model:
            self.video_encoder = AutoModel.from_pretrained(
                config.jepa_encoder_name,
                torch_dtype=self.qwen._get_torch_dtype(config.torch_dtype),
            )
            self.video_processor = AutoVideoProcessor.from_pretrained(config.jepa_encoder_name)
            num_views = config.jepa_tubelet_size
            tubelet_size = self.video_encoder.config.tubelet_size
            image_size = getattr(self.video_encoder.config, "image_size", None)
            if image_size is None:
                first_image_shape = next(iter(config.image_features.values())).shape
                image_size = first_image_shape[-1]
            self.video_predictor = ActionConditionedVideoPredictor(
                num_frames=config.num_video_frames // tubelet_size,
                img_size=(image_size, image_size),
                patch_size=16,
                tubelet_size=1,
                embed_dim=self.video_encoder.config.hidden_size * num_views,
                action_embed_dim=self.qwen.model.config.hidden_size,
                predictor_embed_dim=self.video_encoder.config.hidden_size,
                depth=config.predictor_depth,
                num_heads=config.predictor_num_heads,
                mlp_ratio=config.predictor_mlp_ratio,
                num_action_tokens_per_step=config.num_action_tokens_per_timestep,
            )
        else:
            self.video_encoder = None
            self.video_processor = None
            self.video_predictor = None

        if config.freeze_qwen:
            self.qwen.requires_grad_(False)

        # Build prompt placeholders.
        # Use the encoder's actual tubelet_size when available (world model enabled),
        # otherwise fall back to config.
        _tubelet_size = (
            self.video_encoder.config.tubelet_size
            if config.enable_world_model
            else self.config.jepa_tubelet_size
        )
        num_action_prompt_steps = self.config.num_video_frames // _tubelet_size - 1
        self.replace_prompt = "".join(
            token * self.config.num_action_tokens_per_timestep
            for token in self.action_tokens[:num_action_prompt_steps]
        )
        self.embodied_replace_prompt = (
            self.config.embodied_action_token * self.config.num_embodied_action_tokens_per_instruction
        )

    def _qwen_last_decoder_hidden(self, qwen_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the last decoder hidden state before the final RMSNorm.

        The model was trained with the output of the last transformer block BEFORE
        the final RMSNorm. In transformers 5.x, `hidden_states[-1]` from
        `output_hidden_states=True` is post-norm (tied to `last_hidden_state` via
        `@capture_outputs`). A forward hook on `language_model.layers[-1]` recovers
        the correct pre-RMSNorm state, matching the training-time representation.
        """
        captured: list[torch.Tensor] = []

        def _hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured.append(h)

        last_layer = self.qwen.model.model.language_model.layers[-1]
        handle = last_layer.register_forward_hook(_hook)
        try:
            self.qwen.model(
                **qwen_inputs,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
        finally:
            handle.remove()

        return captured[0]  # [B, seq_len, H]

    # ---- Native VLA-JEPA forward (follows original VLA_JEPA.py) ----

    def _encode_qwen(
        self, images: list[list[Tensor]], instructions: list[str], *, need_action_tokens: bool
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Run Qwen and gather the embodied-action (and optionally action) token hidden states."""
        qwen_inputs = self.qwen.build_inputs(
            images=images,
            instructions=instructions,
            action_prompt=self.replace_prompt,
            embodied_prompt=self.embodied_replace_prompt,
        )
        input_ids = qwen_inputs["input_ids"]
        embodied_idx = (input_ids == self.embodied_action_token_id).nonzero(as_tuple=True)
        action_idx = None
        if need_action_tokens:
            action_mask = torch.isin(input_ids, self._action_token_ids_t)
            action_idx = action_mask.nonzero(as_tuple=True)

        device_type = next(self.parameters()).device.type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            last_hidden = self._qwen_last_decoder_hidden(qwen_inputs)  # [B, seq_len, H]
            b, _, h = last_hidden.shape
            embodied_action_tokens = last_hidden[embodied_idx[0], embodied_idx[1], :].view(b, -1, h)
            action_tokens = (
                last_hidden[action_idx[0], action_idx[1], :].view(b, -1, h)
                if action_idx is not None
                else None
            )
        return embodied_action_tokens, action_tokens

    def _world_model_loss(self, videos: Tensor, action_tokens: Tensor) -> Tensor:
        """JEPA encode + predictor L1 loss. `videos` is [B, V, T, C, H, W] float in [0, 1]."""
        # Match the world model's expected view count: pad with the first view, or trim extras.
        num_views = self.config.jepa_tubelet_size
        if videos.shape[1] < num_views:
            missing = num_views - videos.shape[1]
            videos = torch.cat([videos, videos[:, :1].repeat(1, missing, 1, 1, 1, 1)], dim=1)
        elif videos.shape[1] > num_views:
            videos = videos[:, :num_views]

        b, v, t_frames, c, h_img, w_img = videos.shape
        flat = videos.reshape(b * v, t_frames, c, h_img, w_img)
        # Fast (torchvision) video processor on-device, do_rescale=False (frames already in [0, 1]).
        video_pixels = self.video_processor(
            videos=list(flat),
            return_tensors="pt",
            device=self.video_encoder.device,
            do_rescale=False,
        )["pixel_values_videos"]  # [B*V, T, C, H, W]

        with torch.no_grad():
            video_embeddings = self.video_encoder.get_vision_features(pixel_values_videos=video_pixels)
            # Merge views: [B*V, ...] -> [B, ..., V*embed_dim]
            video_embeddings = torch.cat(torch.chunk(video_embeddings, chunks=v, dim=0), dim=2)

        tubelet_size = self.video_encoder.config.tubelet_size
        # num_video_frames raw frames → t_enc_total temporal positions after tubelet compression
        t_enc_total = self.config.num_video_frames // tubelet_size
        if t_enc_total < 2:
            return torch.zeros((), device=video_embeddings.device)

        # Shift-by-one JEPA split: input_states = positions 0..T-2, gt_states = positions 1..T-1
        t_enc_ctx = t_enc_total - 1
        tokens_per_frame = video_embeddings.shape[1] // t_enc_total
        input_states = video_embeddings[:, : tokens_per_frame * t_enc_ctx, :]
        gt_states = video_embeddings[:, tokens_per_frame:, :]

        expected_actions = t_enc_ctx * self.config.num_action_tokens_per_timestep
        if action_tokens.shape[1] < expected_actions:
            pad = action_tokens[:, -1:].repeat(1, expected_actions - action_tokens.shape[1], 1)
            action_tokens = torch.cat([action_tokens, pad], dim=1)

        predicted_states = self.video_predictor(
            input_states.float(), action_tokens[:, :expected_actions].float()
        )
        return F.l1_loss(predicted_states, gt_states.float(), reduction="mean")

    def _action_loss(
        self,
        embodied_action_tokens: Tensor,
        actions: Tensor,
        state: Tensor | None,
        action_is_pad: Tensor | None,
    ) -> Tensor:
        """Flow-matching action-head loss, repeated over `repeated_diffusion_steps`."""
        device_type = next(self.parameters()).device.type
        with torch.autocast(device_type=device_type, dtype=torch.float32):
            r = self.config.repeated_diffusion_steps
            horizon = self.config.chunk_size
            actions_target = actions[:, -horizon:, :].to(torch.float32).repeat(r, 1, 1)
            embodied = embodied_action_tokens.repeat(r, 1, 1)
            state_rep = state.to(embodied_action_tokens.dtype).repeat(r, 1, 1) if state is not None else None
            pad_rep = action_is_pad[:, -horizon:].repeat(r, 1) if action_is_pad is not None else None
            return self.action_model(embodied, actions_target, state_rep, pad_rep)

    def forward(
        self,
        images: list[list[Tensor]],
        instructions: list[str],
        videos: Tensor | None = None,
        actions: Tensor | None = None,
        state: Tensor | None = None,
        action_is_pad: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Native forward: Qwen encode → optional world-model loss → optional action-head loss."""
        embodied_action_tokens, action_tokens = self._encode_qwen(
            images, instructions, need_action_tokens=self.config.enable_world_model
        )

        if self.config.enable_world_model and videos is not None:
            wm_loss = self._world_model_loss(videos, action_tokens)
        else:
            wm_loss = torch.zeros((), device=embodied_action_tokens.device)

        if actions is None:
            return {"wm_loss": wm_loss}

        action_loss = self._action_loss(embodied_action_tokens, actions, state, action_is_pad)
        return {"action_loss": action_loss, "wm_loss": wm_loss * self.config.world_model_loss_weight}

    # ---- Native predict_action (follows original VLA_JEPA.predict_action) ----

    @torch.no_grad()
    def predict_action(
        self,
        images: list[list[Tensor]],
        instructions: list[str],
        state: Tensor | None = None,
    ) -> Tensor:
        """Predict an action chunk. `images` is per-sample, per-view float [0,1] [C, H, W] tensors."""
        if self.config.resize_images_to is not None:
            height, width = self.config.resize_images_to
            images = [
                [F.interpolate(img[None], size=(height, width), mode="area")[0] for img in views]
                for views in images
            ]

        embodied_action_tokens, _ = self._encode_qwen(images, instructions, need_action_tokens=False)
        return self.action_model.predict_action(
            embodied_action_tokens.float(), state.float() if state is not None else None
        )


# ============================================================================
# LeRobot Adapter Layer - converts between LeRobot batch format and native VLA-JEPA format
# ============================================================================


class VLAJEPAPolicy(PreTrainedPolicy):
    """
    LeRobot adapter for VLA-JEPA.

    Converts LeRobot's standard batch format (dict[str, Tensor]) to the batched tensors
    the native model expects (keeping everything on-device), calls the native model, and
    converts outputs back to LeRobot format.
    """

    config_class = VLAJEPAConfig
    name = "vla_jepa"

    def __init__(self, config: VLAJEPAConfig, **kwargs) -> None:
        super().__init__(config)
        config.validate_features()
        if dataset_meta := kwargs.get("dataset_meta"):
            # cfg.input_features keeps the pretrained model's feature keys (needed for rename_map
            # compatibility), so validate_features() may have read stale dims from a pretrained
            # config. Override state_dim/action_dim from the actual dataset being used.
            ds_features = dataset_meta.features
            if OBS_STATE in ds_features:
                config.state_dim = ds_features[OBS_STATE]["shape"][0]
            if ACTION in ds_features:
                config.action_dim = ds_features[ACTION]["shape"][0]

        self.model = VLAJEPAModel(config)
        self.reset()

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    # ---- Format Conversion: LeRobot → Native ----

    def _prepare_model_inputs(self, batch: dict[str, Tensor], training=True) -> dict[str, Any]:
        """Convert a LeRobot batch to the model's batched, on-device inputs.

        LeRobot format:
            batch = {
                "observation.images.<key>": Tensor [B, C, H, W] or [B, T, C, H, W],
                "observation.state": Tensor [B, state_dim] or [B, T, state_dim],
                "action": Tensor [B, chunk_size, action_dim],  (training only)
                "task": str | List[str],  (optional instruction)
            }

        Returns the kwargs for `VLAJEPAModel.forward` / `.predict_action` (everything stays
        on the batch device; no per-sample shredding): `images` (per-sample, per-view list for
        Qwen messages), `instructions`, and the batched `videos` / `actions` / `state` /
        `action_is_pad` when present.
        """
        image_keys = list(self.config.image_features.keys())
        if not image_keys:
            raise ValueError("VLAJEPA requires at least one image feature.")
        batch_size = batch[image_keys[0]].shape[0]

        # Current-frame image per view ([B, C, H, W]); regroup per sample for Qwen messages.
        frames = []
        for key in image_keys:
            t = batch[key]
            if t.ndim == 5:  # [B, T, C, H, W] -> current observation (delta=0)
                t = t[:, 0]
            frames.append(self.model.qwen.to_pixel_values(t))
        images = [[frame[b] for frame in frames] for b in range(batch_size)]

        tasks = batch.get("task")
        if tasks is None:
            instructions = ["Execute the robot action."] * batch_size
        elif isinstance(tasks, str):
            instructions = [tasks] * batch_size
        else:
            instructions = list(tasks)

        inputs: dict[str, Any] = {"images": images, "instructions": instructions}

        # Videos [B, V, T, C, H, W] - only assembled during training when the world model consumes them.
        if self.model.config.enable_world_model and training:
            views = [batch[k].unsqueeze(1) if batch[k].ndim == 4 else batch[k] for k in image_keys]
            inputs["videos"] = self.model.qwen.to_pixel_values(torch.stack(views, dim=1))

        actions = batch.get(ACTION)
        if actions is not None:
            inputs["actions"] = (actions.unsqueeze(1) if actions.ndim == 2 else actions).float()
            if (pad := batch.get("action_is_pad")) is not None:
                inputs["action_is_pad"] = pad

        state = batch.get(OBS_STATE)
        if state is not None:
            if state.ndim > 2:
                state = state[:, -1, :]
            inputs["state"] = (state.unsqueeze(1) if state.ndim == 2 else state).float()  # [B, 1, dim]

        return inputs

    # ---- LeRobot Policy Interface ----

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """LeRobot train forward: convert → native forward → aggregate losses."""
        native_output = self.model.forward(**self._prepare_model_inputs(batch, training=True))

        ref = next(iter(native_output.values()))
        zero = torch.zeros((), device=ref.device, dtype=ref.dtype)
        total_loss = native_output.get("action_loss", zero) + native_output.get("wm_loss", zero)
        logs = {k: v.detach().item() for k, v in native_output.items()}
        logs["loss"] = total_loss.detach().item()
        return total_loss, logs

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """LeRobot inference: convert → native predict → return as Tensor."""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        inputs = self._prepare_model_inputs(batch, training=False)
        actions = self.model.predict_action(inputs["images"], inputs["instructions"], inputs.get("state"))
        return actions.to(device=self.config.device, dtype=torch.float32)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """LeRobot select_action with action queue caching."""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    @classmethod
    def from_pretrained(
        cls: type[T],
        pretrained_name_or_path: str | Path,
        **kwargs,
    ):
        return super().from_pretrained(pretrained_name_or_path, **kwargs)

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        reinit_prefixes = model.config.reinit_modules
        if not reinit_prefixes:
            return super()._load_as_safetensor(model, model_file, map_location, strict)

        from safetensors.torch import load_file

        state_dict = load_file(model_file, device=map_location)
        current = model.state_dict()

        reinitialized: list[str] = []
        filtered: dict = {}
        for key, value in state_dict.items():
            if key in current and value.shape != current[key].shape:
                if not any(key.startswith(p) for p in reinit_prefixes):
                    raise ValueError(
                        f"Shape mismatch for '{key}' (checkpoint {tuple(value.shape)} vs model "
                        f"{tuple(current[key].shape)}) and its prefix is not in `reinit_modules`."
                    )
                reinitialized.append(
                    f"{key}: checkpoint {tuple(value.shape)} → model {tuple(current[key].shape)}"
                )
            else:
                filtered[key] = value

        if reinitialized:
            logging.warning(
                f"reinit_modules: skipping {len(reinitialized)} tensor(s) with mismatched shapes "
                f"(randomly re-initialised):\n  " + "\n  ".join(reinitialized)
            )

        from lerobot.policies.utils import log_model_loading_keys

        missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)
        log_model_loading_keys(missing_keys, unexpected_keys)
        return model
