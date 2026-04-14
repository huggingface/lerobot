from __future__ import annotations

from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel, AutoVideoProcessor

from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .action_head import VLAJEPAActionHead
from .configuration_vla_jepa import VLAJEPAConfig
from .qwen_interface import Qwen3VLInterface
from .world_model import ActionConditionedVideoPredictor


class VLAJEPAModel(nn.Module):
    def __init__(self, config: VLAJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.qwen = Qwen3VLInterface(config)
        self.action_tokens, self.action_token_ids, self.embodied_action_token_id = self.qwen.expand_tokenizer()
        self.action_model = VLAJEPAActionHead(config, cross_attention_dim=self.qwen.model.config.hidden_size)

        self.video_encoder = AutoModel.from_pretrained(
            config.jepa_encoder_name,
            torch_dtype=self.qwen._get_torch_dtype(config.torch_dtype),
        )
        self.video_processor = AutoVideoProcessor.from_pretrained(config.jepa_encoder_name)
        self.video_predictor = ActionConditionedVideoPredictor(
            embed_dim=self.video_encoder.config.hidden_size,
            action_embed_dim=self.qwen.model.config.hidden_size,
            predictor_embed_dim=self.video_encoder.config.hidden_size,
            depth=config.predictor_depth,
            num_heads=config.predictor_num_heads,
            mlp_ratio=config.predictor_mlp_ratio,
            num_action_tokens_per_step=config.num_action_tokens_per_timestep,
        )
        self.replace_prompt = "".join(
            token * self.config.num_action_tokens_per_timestep
            for token in self.action_tokens[: self.config.num_video_frames - 1]
        )
        self.embodied_replace_prompt = self.config.embodied_action_token * self.config.num_embodied_action_tokens_per_instruction

    def _collect_images(self, batch: dict[str, Tensor]) -> list[list]:
        sample_key = self.config.image_features[0]
        batch_size = batch[sample_key].shape[0]
        images = [[] for _ in range(batch_size)]
        for key in self.config.image_features:
            tensor = batch[key]
            if tensor.ndim == 5:
                tensor = tensor[:, -1]
            for idx in range(batch_size):
                images[idx].append(self.qwen.tensor_to_pil(tensor[idx]))
        return images

    def _collect_videos(self, batch: dict[str, Tensor]) -> torch.Tensor:
        first_key = self.config.image_features[0]
        source = batch[first_key]
        if source.ndim == 4:
            source = source.unsqueeze(1).repeat(1, self.config.num_video_frames, 1, 1, 1)
        elif source.ndim == 5 and source.shape[1] < self.config.num_video_frames:
            pad = source[:, -1:].repeat(1, self.config.num_video_frames - source.shape[1], 1, 1, 1)
            source = torch.cat([source, pad], dim=1)
        elif source.ndim == 5:
            source = source[:, -self.config.num_video_frames :]
        else:
            raise ValueError(f"Unsupported image tensor shape for JEPA: {tuple(source.shape)}")
        return source

    def _get_tasks(self, batch: dict[str, Tensor | list[str] | str]) -> list[str]:
        tasks = batch.get("task")
        if tasks is None:
            return ["Execute the robot action."] * next(iter(batch.values())).shape[0]
        if isinstance(tasks, str):
            return [tasks]
        return list(tasks)

    def _extract_qwen_conditioning(self, batch: dict[str, Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        images = self._collect_images(batch)
        tasks = self._get_tasks(batch)
        qwen_inputs = self.qwen.build_inputs(
            images=images,
            instructions=tasks,
            action_prompt=self.replace_prompt,
            embodied_prompt=self.embodied_replace_prompt,
        )
        outputs = self.qwen.model(
            **qwen_inputs,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        action_mask = torch.isin(
            qwen_inputs["input_ids"],
            torch.tensor(self.action_token_ids, device=qwen_inputs["input_ids"].device),
        )
        action_indices = action_mask.nonzero(as_tuple=True)
        action_tokens = hidden[action_indices[0], action_indices[1], :].view(hidden.shape[0], -1, hidden.shape[-1])

        embodied_mask = qwen_inputs["input_ids"] == self.embodied_action_token_id
        embodied_indices = embodied_mask.nonzero(as_tuple=True)
        embodied_tokens = hidden[embodied_indices[0], embodied_indices[1], :].view(hidden.shape[0], -1, hidden.shape[-1])
        return action_tokens, embodied_tokens

    def _prepare_state(self, batch: dict[str, Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if OBS_STATE not in batch:
            return None
        state = batch[OBS_STATE]
        if state.ndim > 2:
            state = state[:, -1, :]
        return state.to(device=device, dtype=dtype)

    def _prepare_action_targets(self, batch: dict[str, Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        actions = batch[ACTION]
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        horizon = self.config.future_action_window_size + 1
        if actions.shape[1] < horizon:
            pad = actions[:, -1:].repeat(1, horizon - actions.shape[1], 1)
            actions = torch.cat([actions, pad], dim=1)
        return actions[:, -horizon:].to(device=device, dtype=dtype)

    def _encode_video(self, video_tensor: torch.Tensor) -> torch.Tensor:
        processed = []
        for sample in video_tensor:
            processed_sample = self.video_processor(videos=sample, return_tensors="pt")["pixel_values_videos"]
            processed.append(processed_sample)
        pixel_values = torch.cat(processed, dim=0).to(self.video_encoder.device)
        return self.video_encoder.get_vision_features(pixel_values_videos=pixel_values)

    def _compute_world_model_loss(self, batch: dict[str, Tensor], action_tokens: torch.Tensor) -> torch.Tensor | None:
        if not self.config.enable_world_model:
            return None
        video_tensor = self._collect_videos(batch)
        video_features = self._encode_video(video_tensor)
        batch_size = video_tensor.shape[0]
        num_frames = video_tensor.shape[1]
        tokens_per_frame = video_features.shape[1] // num_frames
        video_features = video_features.view(batch_size, num_frames, tokens_per_frame, -1)
        input_states = video_features[:, :-1]
        gt_states = video_features[:, 1:]

        expected_tokens = (num_frames - 1) * self.config.num_action_tokens_per_timestep
        if action_tokens.shape[1] < expected_tokens:
            pad = action_tokens[:, -1:].repeat(1, expected_tokens - action_tokens.shape[1], 1)
            action_tokens = torch.cat([action_tokens, pad], dim=1)
        action_tokens = action_tokens[:, :expected_tokens]
        action_tokens = action_tokens.view(batch_size, num_frames - 1, self.config.num_action_tokens_per_timestep, -1)
        pred_states = self.video_predictor(input_states, action_tokens)
        return F.l1_loss(pred_states, gt_states, reduction="mean")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        action_tokens, embodied_tokens = self._extract_qwen_conditioning(batch)
        state = self._prepare_state(batch, embodied_tokens.device, embodied_tokens.dtype)
        target_actions = self._prepare_action_targets(batch, embodied_tokens.device, embodied_tokens.dtype)
        action_loss = self.action_model(embodied_tokens, target_actions, state)

        wm_loss = self._compute_world_model_loss(batch, action_tokens)
        total_loss = action_loss
        logs = {"action_loss": action_loss.detach()}
        if wm_loss is not None:
            total_loss = total_loss + self.config.world_model_loss_weight * wm_loss
            logs["wm_loss"] = wm_loss.detach()
        logs["loss"] = total_loss.detach()
        return total_loss, logs

    @torch.no_grad()
    def predict_action(self, batch: dict[str, Tensor]) -> Tensor:
        _, embodied_tokens = self._extract_qwen_conditioning(batch)
        state = self._prepare_state(batch, embodied_tokens.device, embodied_tokens.dtype)
        return self.action_model.predict_action(embodied_tokens, state)


class VLAJEPAPolicy(PreTrainedPolicy):
    config_class = VLAJEPAConfig
    name = "vla_jepa"

    def __init__(self, config: VLAJEPAConfig, **kwargs) -> None:
        super().__init__(config)
        config.validate_features()
        self.model = VLAJEPAModel(config)
        self.reset()

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        loss, logs = self.model(batch)
        return loss, {key: value.item() for key, value in logs.items()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:  # noqa: ARG002
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        return self.model.predict_action(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:  # noqa: ARG002
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if len(self._queues[ACTION]) == 0:
            actions = self.model.predict_action(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    @classmethod
    def from_pretrained(
        cls: type[T],
        pretrained_name_or_path: str | Path,
        **kwargs,
    ):
        return super().from_pretrained(pretrained_name_or_path, **kwargs)
