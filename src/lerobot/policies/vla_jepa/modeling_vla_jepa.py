from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
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

    Input: List[dict] native format (same as original starVLA)
      - "image": List[PIL.Image] (multi-view images)
      - "video": np.ndarray [V, T, H, W, 3]
      - "lang": str (task instruction)
      - "action": np.ndarray [T, action_dim] (optional, training only)
      - "state": np.ndarray [1, state_dim] (optional)
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

        # Action head (flow-matching DiT)
        self.action_model = VLAJEPAActionHead(config, cross_attention_dim=self.qwen.model.config.hidden_size)

        # JEPA world model components
        if config.enable_world_model:
            self.video_encoder = AutoModel.from_pretrained(
                config.jepa_encoder_name,
                torch_dtype=self.qwen._get_torch_dtype(config.torch_dtype),
            )
            self.video_processor = AutoVideoProcessor.from_pretrained(config.jepa_encoder_name)
            num_views = max(len(config.image_features), 1)
            self.video_predictor = ActionConditionedVideoPredictor(
                embed_dim=num_views * self.video_encoder.config.hidden_size,
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
        # Original uses num_frames // tubelet_size - 1 action token groups for the world model predictor.
        # This matches the number of context temporal positions after tubelet compression.
        n_wm_action_groups = max(1, self.config.num_video_frames // self.config.jepa_tubelet_size - 1)
        self.replace_prompt = "".join(
            token * self.config.num_action_tokens_per_timestep
            for token in self.action_tokens[:n_wm_action_groups]
        )
        self.embodied_replace_prompt = (
            self.config.embodied_action_token * self.config.num_embodied_action_tokens_per_instruction
        )

    # ---- Native VLA-JEPA forward (follows original VLA_JEPA.py) ----

    def forward(self, examples: list[dict]) -> dict[str, Tensor]:
        """
        Native forward pass following original starVLA VLA_JEPA.forward.

        Args:
            examples: List of per-sample dicts with keys:
                "image"  : List[PIL.Image]  — multi-view images
                "video"  : np.ndarray [V, T, H, W, 3]
                "lang"   : str — task instruction
                "action" : np.ndarray [T, action_dim] (optional)
                "state"  : np.ndarray [1, state_dim] (optional)

        Returns:
            dict with "action_loss" and "wm_loss" keys (scalar Tensors).
        """
        # Unpack native format (same pattern as original VLA_JEPA.py)
        batch_images = [ex["image"] for ex in examples]  # List[List[PIL.Image]]
        batch_videos = [ex["video"] for ex in examples]  # List[np.ndarray]
        instructions = [ex["lang"] for ex in examples]  # List[str]
        has_action = "action" in examples[0] and examples[0]["action"] is not None
        actions = [ex["action"] for ex in examples] if has_action else None
        has_state = "state" in examples[0] and examples[0]["state"] is not None
        state = [ex["state"] for ex in examples] if has_state else None
        action_is_pad = (
            [ex["action_is_pad"] for ex in examples]
            if has_action and "action_is_pad" in examples[0] and examples[0]["action_is_pad"] is not None
            else None
        )

        # Stack videos: [B, V, T, H, W, 3] -> [B, V, T, 3, H, W]
        batch_videos = np.stack(batch_videos)
        batch_videos = batch_videos.transpose(0, 1, 2, 5, 3, 4)  # [B, V, T, 3, H, W]

        # ---- Step 1: QwenVL encode (same as original) ----
        qwen_inputs = self.qwen.build_inputs(
            images=batch_images,
            instructions=instructions,
            action_prompt=self.replace_prompt,
            embodied_prompt=self.embodied_replace_prompt,
        )

        # Locate embodied-action tokens (always needed for action head)
        embodied_mask = qwen_inputs["input_ids"] == self.embodied_action_token_id
        embodied_indices = embodied_mask.nonzero(as_tuple=True)

        # Locate action tokens (only needed for world model predictor)
        if self.config.enable_world_model:
            action_mask = torch.isin(
                qwen_inputs["input_ids"],
                torch.tensor(self.action_token_ids, device=qwen_inputs["input_ids"].device),
            )
            action_indices = action_mask.nonzero(as_tuple=True)

        device_type = next(self.parameters()).device.type

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            qwen_outputs = self.qwen.model(
                **qwen_inputs,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
            )
            last_hidden = qwen_outputs.hidden_states[-1]  # [B, seq_len, H]
            b, _, h = last_hidden.shape

            if self.config.enable_world_model:
                action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(b, -1, h)

            embodied_action_tokens = last_hidden[embodied_indices[0], embodied_indices[1], :].view(b, -1, h)

        # ---- Step 2+3: JEPA Encoder + Predictor ----
        device_wm = last_hidden.device
        if not self.config.enable_world_model:
            wm_loss = torch.tensor(0.0, device=device_wm)
        else:
            b, v, t_frames, c, h_img, w_img = batch_videos.shape
            batch_videos_flat = batch_videos.reshape(b * v, t_frames, c, h_img, w_img)

            video_pixels = []
            for i in range(b * v):
                video_pixels.append(
                    self.video_processor(videos=batch_videos_flat[i], return_tensors="pt")[
                        "pixel_values_videos"
                    ].to(self.video_encoder.device)
                )
            video_pixels = torch.cat(video_pixels, dim=0)  # [B*V, T, C, H, W]

            with torch.no_grad():
                video_embeddings = self.video_encoder.get_vision_features(pixel_values_videos=video_pixels)
                # Merge views: [B*V, ...] -> [B, ..., V*embed_dim]
                video_embeddings = torch.cat(torch.chunk(video_embeddings, chunks=v, dim=0), dim=2)

            tubelet_size = self.video_encoder.config.tubelet_size
            device_wm = video_embeddings.device
            # num_video_frames raw frames → t_enc_total temporal positions after tubelet compression
            t_enc_total = self.config.num_video_frames // tubelet_size

            if t_enc_total < 2:
                wm_loss = torch.tensor(0.0, device=device_wm)
            else:
                # Shift-by-one JEPA split (matches original VLA_JEPA.py lines 231-232):
                # input_states: positions 0..T-2, gt_states: positions 1..T-1
                t_enc_ctx = t_enc_total - 1
                tokens_per_frame = video_embeddings.shape[1] // t_enc_total

                input_states = video_embeddings[:, : tokens_per_frame * t_enc_ctx, :]
                gt_states = video_embeddings[:, tokens_per_frame:, :]
                d_emb = input_states.shape[-1]

                input_states_4d = input_states.view(b, t_enc_ctx, tokens_per_frame, d_emb)

                expected_actions = t_enc_ctx * self.config.num_action_tokens_per_timestep
                if action_tokens.shape[1] < expected_actions:
                    pad = action_tokens[:, -1:].repeat(1, expected_actions - action_tokens.shape[1], 1)
                    action_tokens = torch.cat([action_tokens, pad], dim=1)
                act_4d = action_tokens[:, :expected_actions].view(
                    b, t_enc_ctx, self.config.num_action_tokens_per_timestep, -1
                )

                pred_4d = self.video_predictor(input_states_4d.float(), act_4d.float())
                predicted_states = pred_4d.reshape(b, -1, d_emb)

                wm_loss = F.l1_loss(predicted_states, gt_states.float(), reduction="mean")

        if not has_action:
            return {"wm_loss": wm_loss}

        # ---- Step 4: Action Head ----
        with torch.autocast(device_type=device_type, dtype=torch.float32):
            actions_tensor = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=torch.float32
            )  # [B, T_full, action_dim]
            action_horizon = self.config.future_action_window_size + 1
            actions_target = actions_tensor[:, -action_horizon:, :]

            state_tensor = None
            if state is not None:
                state_tensor = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=torch.float32
                )  # [B, 1, state_dim]

            # repeated_diffusion_steps: draw R independent noise samples per batch item (CogACT-style).
            # Effectively multiplies data efficiency of the action head by R with no extra Qwen/JEPA cost.
            num_repeated = self.config.repeated_diffusion_steps
            embodied_rep = embodied_action_tokens.float().repeat(num_repeated, 1, 1)
            actions_rep = actions_target.repeat(num_repeated, 1, 1)
            state_rep = state_tensor.repeat(num_repeated, 1, 1) if state_tensor is not None else None

            action_is_pad_rep = None
            if action_is_pad is not None:
                pad_tensor = torch.stack(
                    [
                        p.to(actions_target.device)
                        if isinstance(p, Tensor)
                        else torch.tensor(p, device=actions_target.device)
                        for p in action_is_pad
                    ]
                )  # [B, T_full]
                pad_tensor = pad_tensor[:, -action_horizon:]  # [B, action_horizon]
                action_is_pad_rep = pad_tensor.repeat(num_repeated, 1)  # [B*R, action_horizon]

            action_loss = self.action_model(embodied_rep, actions_rep, state_rep, action_is_pad_rep)

        return {"action_loss": action_loss, "wm_loss": wm_loss * self.config.world_model_loss_weight}

    # ---- Native predict_action (follows original VLA_JEPA.predict_action) ----

    @torch.no_grad()
    def predict_action(
        self,
        batch_images: list[list[Image.Image]],
        instructions: list[str],
        state: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Native action prediction following original VLA_JEPA.predict_action.

        Args:
            batch_images: List of samples; each is List[PIL.Image] (multi-view).
            instructions: Task instructions, one per sample.
            state: Optional [B, state_dim] numpy array.

        Returns:
            np.ndarray [B, action_horizon, action_dim] — predicted actions.
        """
        qwen_inputs = self.qwen.build_inputs(
            images=batch_images,
            instructions=instructions,
            action_prompt=self.replace_prompt,
            embodied_prompt=self.embodied_replace_prompt,
        )

        embodied_mask = qwen_inputs["input_ids"] == self.embodied_action_token_id
        embodied_indices = embodied_mask.nonzero(as_tuple=True)

        device_type = next(self.parameters()).device.type

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            qwen_outputs = self.qwen.model(
                **qwen_inputs,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
            )
            last_hidden = qwen_outputs.hidden_states[-1]
            b, _, h = last_hidden.shape
            embodied_action_tokens = last_hidden[embodied_indices[0], embodied_indices[1], :].view(b, -1, h)

        state_tensor = None
        if state is not None:
            state_tensor = torch.from_numpy(np.array(state)).to(
                device=last_hidden.device, dtype=torch.float32
            )

        with torch.autocast(device_type=device_type, dtype=torch.float32):
            # Cast embodied tokens to float32 for action model compatibility
            pred_actions = self.action_model.predict_action(
                embodied_action_tokens.float(), state_tensor
            )  # [B, action_horizon, action_dim]

        return pred_actions.detach().cpu().numpy()


# ============================================================================
# LeRobot Adapter Layer - converts between LeRobot batch format and native VLA-JEPA format
# ============================================================================


class VLAJEPAPolicy(PreTrainedPolicy):
    """
    LeRobot adapter for VLA-JEPA.

    Converts LeRobot's standard batch format (dict[str, Tensor]) to the native
    VLA-JEPA format (List[dict]), calls the native model, and converts outputs
    back to LeRobot format.
    """

    config_class = VLAJEPAConfig
    name = "vla_jepa"

    def __init__(self, config: VLAJEPAConfig, **kwargs) -> None:
        super().__init__(config)
        config.validate_features()
        self.model = VLAJEPAModel(config)
        self.reset()

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    # ---- Format Conversion: LeRobot → Native ----

    def _lerobot_to_native(self, batch: dict[str, Tensor]) -> list[dict]:
        """
        Convert LeRobot batch format to native VLA-JEPA examples format.

        LeRobot format:
            batch = {
                "observation.images.<key>": Tensor [B, C, H, W] or [B, T, C, H, W],
                "observation.state": Tensor [B, state_dim] or [B, T, state_dim],
                "action": Tensor [B, chunk_size, action_dim],  (training only)
                "task": str | List[str],  (optional instruction)
            }

        Native format (List[dict]):
            {
                "image": List[PIL.Image],       # multi-view images per sample
                "video": np.ndarray [V, T, H, W, 3],
                "lang": str,                     # task instruction
                "action": np.ndarray [T, action_dim],  # optional
                "state": np.ndarray [1, state_dim],    # optional
            }
        """
        # Determine batch size from the first image feature
        image_keys = list(self.config.image_features.keys())
        if not image_keys:
            raise ValueError("VLAJEPA requires at least one image feature.")
        first_key = image_keys[0]
        first_tensor = batch[first_key]
        batch_size = first_tensor.shape[0]

        # ---- Collect images per sample ----
        # images_per_sample[b][v] = PIL.Image for view v
        images_per_sample: list[list[Image.Image]] = [[] for _ in range(batch_size)]
        for key in image_keys:
            tensor = batch[key]  # [B, C, H, W] or [B, T, C, H, W]
            if tensor.ndim == 5:
                # observation_delta_indices = [0, 1, ..., num_video_frames-1]
                # index 0 is the current observation (delta=0)
                tensor = tensor[:, 0]
            for b in range(batch_size):
                images_per_sample[b].append(self.model.qwen.tensor_to_pil(tensor[b]))

        # ---- Collect videos per sample ----
        # Build video arrays: for each sample, stack views as [V, T, H, W, 3]
        # Check whether any image feature has a time dimension
        video_source = None
        for k in image_keys:
            if k in batch:
                video_source = batch[k]  # Use first available for shape inspection
                break

        if video_source is None:
            raise ValueError("No image data found in batch for video construction.")

        videos_per_sample = []
        for b in range(batch_size):
            sample_views = []
            for k in image_keys:
                t = batch[k][b]  # [C, H, W] or [T, C, H, W]
                if t.ndim == 3:
                    t = t.unsqueeze(0)  # [1, C, H, W]
                # Convert to [T, H, W, 3] numpy
                t_np = t.permute(0, 2, 3, 1).detach().cpu().float().numpy()
                # Clamp to [0, 255]
                if t_np.max() <= 1.0:
                    t_np = t_np * 255.0
                t_np = t_np.clip(0, 255).astype(np.uint8)
                sample_views.append(t_np)
            # Stack views: [V, T, H, W, 3]
            videos_per_sample.append(np.stack(sample_views, axis=0))

        # ---- Collect instructions ----
        tasks = batch.get("task")
        if tasks is None:
            instructions = ["Execute the robot action."] * batch_size
        elif isinstance(tasks, str):
            instructions = [tasks] * batch_size
        else:
            instructions = list(tasks)

        # ---- Collect actions (training only) ----
        actions_list = None
        action_is_pad_list = None
        actions_tensor = batch.get(ACTION)
        if actions_tensor is not None:
            if actions_tensor.ndim == 2:
                actions_tensor = actions_tensor.unsqueeze(1)
            actions_list = [actions_tensor[b].detach().cpu().float().numpy() for b in range(batch_size)]
            action_is_pad_tensor = batch.get("action_is_pad")
            if action_is_pad_tensor is not None:
                action_is_pad_list = [action_is_pad_tensor[b].detach().cpu() for b in range(batch_size)]

        # ---- Collect state ----
        state_list = None
        state_tensor = batch.get(OBS_STATE)
        if state_tensor is not None:
            if state_tensor.ndim > 2:
                state_tensor = state_tensor[:, -1, :]
            if state_tensor.ndim == 2:
                state_tensor = state_tensor.unsqueeze(1)  # [B, 1, state_dim]
            state_list = [state_tensor[b].detach().cpu().float().numpy() for b in range(batch_size)]

        # ---- Assemble native examples ----
        examples = []
        for b in range(batch_size):
            example = {
                "image": images_per_sample[b],
                "video": videos_per_sample[b],
                "lang": instructions[b],
            }
            if actions_list is not None:
                example["action"] = actions_list[b]
            if action_is_pad_list is not None:
                example["action_is_pad"] = action_is_pad_list[b]
            if state_list is not None:
                example["state"] = state_list[b]
            examples.append(example)

        return examples

    # ---- Format Conversion: Native → LeRobot ----

    def _native_to_lerobot(self, native_output: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """
        Convert native VLA-JEPA output dict to LeRobot (loss, logs) format.

        Native output:
            {"action_loss": Tensor, "wm_loss": Tensor}
            or {"wm_loss": Tensor}  (video-only mode)

        LeRobot output:
            (total_loss: scalar Tensor, {"action_loss": float, "wm_loss": float, "loss": float})
        """
        logs: dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=self.config.device)

        if "action_loss" in native_output:
            total_loss = total_loss + native_output["action_loss"]
            logs["action_loss"] = native_output["action_loss"].detach().item()

        if "wm_loss" in native_output:
            total_loss = total_loss + native_output["wm_loss"]
            logs["wm_loss"] = native_output["wm_loss"].detach().item()

        logs["loss"] = (
            total_loss.detach().item()
            if total_loss.item() != 0
            else (logs.get("wm_loss", 0.0) + logs.get("action_loss", 0.0))
        )

        return total_loss, logs

    # ---- LeRobot Policy Interface ----

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """LeRobot train forward: convert → native forward → convert back."""
        examples = self._lerobot_to_native(batch)
        native_output = self.model.forward(examples)
        return self._native_to_lerobot(native_output)

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """LeRobot inference: convert → native predict → return as Tensor."""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # Convert to native format
        examples = self._lerobot_to_native(batch)
        batch_images = [ex["image"] for ex in examples]
        instructions = [ex["lang"] for ex in examples]

        state_np = None
        if "state" in examples[0] and examples[0]["state"] is not None:
            state_np = np.stack([ex["state"] for ex in examples])

        # Call native predict
        actions_np = self.model.predict_action(batch_images, instructions, state_np)

        # Convert back to tensor on the right device
        return torch.from_numpy(actions_np).to(device=self.config.device, dtype=torch.float32)

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
