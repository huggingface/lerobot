# Copyright 2026 Shirui Chen, Cole Harrison, Ying-Chun Lee, Angela Jin Yang,
# Zhongzheng Ren, Lillian J. Ratliff, Jiafei Duan, Dieter Fox, Ranjay Krishna
# and The HuggingFace Inc. team. All rights reserved.
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

"""TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics.

Paper:         https://arxiv.org/abs/2602.19313
Project:       https://topreward.github.io/webpage/
Original code: https://github.com/TOPReward/TOPReward
Backbone:      https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct  (default)

TOPReward is a **zero-shot** reward model: it has no fine-tuned weights of
its own. Given a video trajectory and a task instruction, it asks an
off-the-shelf VLM how likely the instruction is, conditioned on the video,
and returns that log-likelihood as the reward signal.

Inference recipe:

1. Build a chat-style prompt:
   ``[video(frames, fps), text=prompt_prefix, text="{instruction} ... True"]``
2. Forward the full token sequence through the VLM.
3. Mask all but the final token with ``-100`` (``prompt_length = input_len - 1``,
   mirrored from upstream). After the standard causal-LM next-token shift, this
   isolates the single position where the model predicts the literal ``"True"``
   that ends the prompt — the binary "is the instruction true given the video?"
   answer.
4. Read that token's log-probability from the logits and reduce it (mean or sum
   — equivalent for a single token, kept for API parity with upstream) into a
   scalar reward.

This LeRobot port is **inference-only and not trainable** — :meth:`forward`
is intentionally inherited from :class:`PreTrainedRewardModel` and raises
``NotImplementedError``, making :attr:`PreTrainedRewardModel.is_trainable`
return ``False``.

Because the VLM weights live on the Hugging Face Hub under their canonical
id (``Qwen/Qwen3-VL-8B-Instruct`` etc.) and TOPReward never modifies them,
:meth:`_save_pretrained` and :meth:`from_pretrained` are overridden so a
TOPReward LeRobot "checkpoint" is a single ``config.json`` (the VLM is
re-fetched from the Hub at load time).
"""

from __future__ import annotations

import builtins
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image
from torch import Tensor

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.rewards.topreward.configuration_topreward import TOPRewardConfig
from lerobot.rewards.topreward.processor_topreward import TOPREWARD_FEATURE_PREFIX
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
else:
    AutoProcessor = None  # type: ignore[assignment]
    Qwen3VLForConditionalGeneration = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="TOPRewardModel")

_TRUE_ANSWER = "True"


def _torch_dtype(name: str) -> torch.dtype | str:
    """Resolve a torch dtype name; ``"auto"`` is passed through verbatim."""
    if name == "auto":
        return "auto"
    dtype = getattr(torch, name, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    raise ValueError(f"Unknown torch dtype: {name!r}")


def _frames_to_pil(frames: np.ndarray) -> list[Image.Image]:
    """Convert ``(T, H, W, C)`` uint8 frames to a list of PIL images."""
    if frames.ndim != 4:
        raise ValueError(f"Expected (T,H,W,C) frames; got shape {frames.shape}")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return [Image.fromarray(frames[i]) for i in range(frames.shape[0])]


def minmax_normalize_rewards(rewards: list[float] | np.ndarray) -> np.ndarray:
    """Min-max normalise raw log-prob rewards into ``[0, 1]``.

    Matches upstream ``QwenClient.normalize_rewards(rewards, method="minmax")``:
    a single-element input maps to ``[1.0]`` (no information to scale), and a
    flat input (``max == min``) maps to all-ones.
    """
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    if rewards_arr.size == 0:
        return rewards_arr.astype(np.float32)
    if rewards_arr.size == 1:
        return np.array([1.0], dtype=np.float32)
    r_min, r_max = rewards_arr.min(), rewards_arr.max()
    if r_max == r_min:
        return np.ones_like(rewards_arr, dtype=np.float32)
    return ((rewards_arr - r_min) / (r_max - r_min)).astype(np.float32)


class TOPRewardModel(PreTrainedRewardModel):
    """TOPReward zero-shot reward model."""

    name = "topreward"
    config_class = TOPRewardConfig

    def __init__(self, config: TOPRewardConfig) -> None:
        require_package("transformers", extra="topreward")
        require_package("qwen-vl-utils", extra="topreward", import_name="qwen_vl_utils")
        super().__init__(config)
        self.config = config

        torch_dtype = _torch_dtype(config.torch_dtype)
        model_kwargs: dict[str, Any] = {"dtype": torch_dtype, "trust_remote_code": True}
        if config.attn_implementation is not None:
            model_kwargs["attn_implementation"] = config.attn_implementation

        # TOPReward is zero-shot: load the VLM as-is from the Hub. No
        # weights of our own, no embedding resize, no head wiring.
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(config.vlm_name, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(config.vlm_name, trust_remote_code=True)

    def compute_reward(self, batch: dict[str, Any]) -> Tensor:
        """Return one log-prob reward per sample in the batch.

        Expects a batch produced by :class:`TOPRewardEncoderProcessorStep`:
        ``observation[f"{TOPREWARD_FEATURE_PREFIX}frames"]`` is a list of
        ``(T, H, W, C) uint8`` numpy arrays (one per sample) and
        ``observation[f"{TOPREWARD_FEATURE_PREFIX}task"]`` is a list of
        task strings of the same length.
        """
        frames_per_sample, tasks = self._unpack_batch(batch)
        rewards = [
            self._compute_log_prob_reward(frames, task)
            for frames, task in zip(frames_per_sample, tasks, strict=True)
        ]
        out = torch.as_tensor(rewards, dtype=torch.float32)
        if np.isfinite(self.config.success_threshold):
            out = (out > self.config.success_threshold).float()
        return out.to(self.config.device or "cpu")

    @torch.no_grad()
    def predict_curves(
        self,
        batch: dict[str, Any],
        *,
        num_prefixes: int | None = None,
    ) -> dict[str, Tensor]:
        """Per-sample dense progress curves over prefixes ``[0, t]``.

        Mirrors upstream ``compute_instruction_rewards_for_prefixes``: for
        each sample we run one VLM forward per prefix length and read the
        log-prob reward at that prefix. Raw log-probs are then min-max
        normalised per-trajectory to ``[0, 1]``. Because trajectories
        within a batch can have different lengths, the returned
        ``progress`` tensor is right-padded with ``NaN`` to the longest
        trajectory in the batch.

        Args:
            batch: Same input as :meth:`compute_reward`.
            num_prefixes: How many evenly-spaced prefix lengths to score
                per trajectory. ``None`` (default) uses every prefix
                length ``[1, N]`` → fully dense, ``N`` VLM forwards per
                trajectory. Pass a smaller integer (e.g. ``15``, the
                upstream default) for sparse-dense scoring with linear
                interpolation between anchors.

        Returns:
            Dict with one float32 CPU tensor:

            - ``progress``: ``(B, T_max)`` — per-frame progress in
              ``[0, 1]`` (min-max normalised log-prob curve), padded with
              ``NaN``.
        """
        if num_prefixes is not None and num_prefixes < 1:
            raise ValueError(f"num_prefixes must be >= 1 or None, got {num_prefixes}")

        frames_per_sample, tasks = self._unpack_batch(batch)
        curves: list[np.ndarray] = []
        max_len = 0
        for frames, task in zip(frames_per_sample, tasks, strict=True):
            num_frames = int(frames.shape[0])
            if num_frames == 0:
                curves.append(np.zeros(0, dtype=np.float32))
                continue

            if num_prefixes is None or num_prefixes >= num_frames:
                anchor_lengths = np.arange(1, num_frames + 1, dtype=np.int64)
            else:
                # Match upstream: linspace from 1 to N, dedupe (rounding
                # collisions for short trajectories), sort ascending.
                anchor_lengths = np.unique(np.linspace(1, num_frames, num_prefixes).round().astype(np.int64))

            raw_rewards = [self._compute_log_prob_reward(frames[:length], task) for length in anchor_lengths]
            normalized_at_anchors = minmax_normalize_rewards(raw_rewards)

            # Linear interpolation back to per-frame resolution when
            # `num_prefixes < num_frames`.
            if anchor_lengths.shape[0] == num_frames:
                per_frame = normalized_at_anchors
            else:
                per_frame = np.interp(
                    np.arange(1, num_frames + 1, dtype=np.float64),
                    anchor_lengths.astype(np.float64),
                    normalized_at_anchors.astype(np.float64),
                ).astype(np.float32)

            curves.append(per_frame)
            max_len = max(max_len, num_frames)

        padded = np.full((len(curves), max_len), np.nan, dtype=np.float32)
        for i, curve in enumerate(curves):
            padded[i, : curve.shape[0]] = curve
        return {"progress": torch.from_numpy(padded)}

    # ------------------------------------------------------------------
    # Save / load — VLM weights are not stored in our checkpoint
    # ------------------------------------------------------------------

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save ``config.json`` only.

        TOPReward has no fine-tuned weights of its own — the VLM is
        identified by :attr:`TOPRewardConfig.vlm_name` and lives on the
        Hugging Face Hub under that id. Writing the VLM into a
        ``model.safetensors`` here would just duplicate ~16 GB of Qwen
        weights under our org for no benefit.
        """
        self.config._save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: RewardModelConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,  # accepted for API parity; unused
        **kwargs: Any,
    ) -> T:
        """Load a TOPReward configuration and instantiate the wrapped VLM.

        Two modes:

        - Local directory containing ``config.json``: read the config and
          rebuild the model. The VLM is re-fetched from the Hub via
          :attr:`TOPRewardConfig.vlm_name`.
        - HF Hub repo id: download just ``config.json``, same as above.
        """
        del strict  # TOPReward has no weights of its own to (strictly) load.
        if config is None:
            config = RewardModelConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        if not isinstance(config, TOPRewardConfig):
            raise TypeError(
                f"Expected a TOPRewardConfig, got {type(config).__name__}. Make sure "
                f"`pretrained_name_or_path={pretrained_name_or_path!r}` points at a "
                "TOPReward checkpoint."
            )

        model_id = str(pretrained_name_or_path)
        if not os.path.isdir(model_id):
            # Validate that the remote repo at least contains a TOPReward config.json
            try:
                hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        instance = cls(config, **kwargs)
        instance.to(config.device)
        instance.eval()
        return instance

    def push_model_to_hub(self, cfg: TrainPipelineConfig):
        """Push the TOPReward ``config.json`` + model card to the Hub.

        Skips the safetensors upload — the wrapped VLM is identified by
        ``vlm_name`` and we never modify it.
        """
        api = HfApi()
        repo_id = api.create_repo(
            repo_id=self.config.repo_id, private=self.config.private, exist_ok=True
        ).repo_id

        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            saved_path = Path(tmp) / repo_id
            saved_path.mkdir(parents=True, exist_ok=True)

            self.config._save_pretrained(saved_path)

            card = self.generate_model_card(
                cfg.dataset.repo_id, self.config.type, self.config.license, self.config.tags
            )
            card.save(str(saved_path / "README.md"))

            cfg.save_pretrained(saved_path)

            commit_info = api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message="Upload TOPReward config and readme",
                allow_patterns=["*.json", "*.yaml", "*.md"],
                ignore_patterns=["*.tmp", "*.log", "*.safetensors"],
            )

            logger.info(f"Model pushed to {commit_info.repo_url.url}")

    def _unpack_batch(self, batch: dict[str, Any]) -> tuple[list[np.ndarray], list[str]]:
        frames_key = f"{TOPREWARD_FEATURE_PREFIX}frames"
        task_key = f"{TOPREWARD_FEATURE_PREFIX}task"
        if frames_key not in batch or task_key not in batch:
            raise KeyError(
                "TOPReward batch missing pre-encoded inputs (expected "
                f"`{frames_key}` and `{task_key}`). Make sure the "
                "TOPRewardEncoderProcessorStep ran before `compute_reward`."
            )
        frames_per_sample = list(batch[frames_key])
        tasks = list(batch[task_key])
        if len(frames_per_sample) != len(tasks):
            raise ValueError(
                f"frames batch size ({len(frames_per_sample)}) does not match task batch size ({len(tasks)})"
            )
        return frames_per_sample, tasks

    @torch.no_grad()
    def _compute_log_prob_reward(self, frames: np.ndarray, instruction: str) -> float:
        """Compute the log-likelihood of the final answer token given the prompt.

        Port of ``QwenClient.compute_instruction_reward`` (the upstream
        TOPReward implementation), stripped of the
        :class:`InstructionRewardResult` metadata wrapper we don't need.
        Returns ``log P(final_token | video + prompt + instruction)`` — by
        default the final token is the literal ``"True"`` that closes the
        suffix template, which is the binary "is the instruction satisfied"
        signal the paper describes.
        """
        device = next(self.model.parameters()).device
        pil_frames = _frames_to_pil(frames)

        if self.config.use_video_description:
            description = self._generate_object_state_reasoning(pil_frames)
            prompt_text = (
                f"{description} Therefore given the above description and the "
                "video, the video shows a robot manipulation trajectory that "
                "**completes** the following instruction: "
            )
        else:
            prompt_text = self.config.prompt_prefix

        eos_token = self.processor.tokenizer.eos_token
        instruction_suffix = self.config.prompt_suffix_template.format(instruction=instruction)

        # Two prompt assembly modes match the upstream:
        #
        # - ``add_chat_template=True``: wrap the FULL prompt (including
        #   instruction) with the chat template, then append the literal
        #   ``"True"`` token outside the template.
        # - ``add_chat_template=False``: apply the chat template to the
        #   video+prefix only (no generation prompt), strip the trailing
        #   EOS, then concatenate the literal instruction suffix.
        if self.config.add_chat_template:
            # Suffix excluding the trailing "True" — we want "True" to be
            # the scored token, not part of the template's user turn.
            suffix_for_template = instruction_suffix.removesuffix(_TRUE_ANSWER).rstrip()
            templated_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": pil_frames, "fps": self.config.fps},
                        {"type": "text", "text": f"{prompt_text}{suffix_for_template}"},
                    ],
                }
            ]
            prompt_chat = self.processor.apply_chat_template(
                templated_messages, tokenize=False, add_generation_prompt=True
            )
            full_text = f"{prompt_chat}{_TRUE_ANSWER}"
            image_inputs, video_inputs = self._process_vision_info(templated_messages)
        else:
            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": pil_frames, "fps": self.config.fps},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            prompt_chat = self.processor.apply_chat_template(
                user_messages, tokenize=False, add_generation_prompt=False
            )
            if eos_token is not None:
                prompt_chat = prompt_chat.split(eos_token)[0]
            full_text = f"{prompt_chat}{instruction_suffix}"
            image_inputs, video_inputs = self._process_vision_info(user_messages)

        inputs = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        input_len = int(inputs["input_ids"].shape[-1])
        if input_len > self.config.max_input_length:
            raise ValueError(
                f"TOPReward input length {input_len} exceeds max_input_length "
                f"{self.config.max_input_length}; lower `max_frames` or raise `max_input_length`."
            )

        labels = inputs["input_ids"].clone()
        # Mask everything except the very last token. ``prompt_length = input_len - 1``
        # mirrors upstream ``QwenClient.compute_instruction_reward``; after the
        # causal-LM next-token shift below this isolates exactly one position —
        # the prediction of the literal ``"True"`` that closes ``prompt_suffix_template``.
        # The resulting reward is therefore ``log P("True" | video + prompt + instruction)``.
        prompt_length = input_len - 1
        labels[:, :prompt_length] = -100
        if "attention_mask" in inputs:
            labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

        self.model.eval()
        outputs = self.model(**inputs, labels=labels)

        logits = outputs.logits[:, :-1, :]
        target_labels = labels[:, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        mask = target_labels != -100
        safe_targets = target_labels.masked_fill(~mask, 0)
        token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        masked_log_probs = token_log_probs[mask]
        if masked_log_probs.numel() == 0:
            raise RuntimeError(
                "TOPReward could not isolate any suffix tokens to score. Check that "
                "`prompt_suffix_template` produces at least one tokenised character."
            )

        # ``mean`` vs ``sum`` are equivalent for a single scored token but the
        # knob is kept for API parity with upstream (and for forward-compat with
        # any future variant that scores more than the final answer token).
        if self.config.reduction == "sum":
            reward = masked_log_probs.sum().item()
        else:  # mean
            reward = masked_log_probs.mean().item()
        return float(reward)

    @torch.no_grad()
    def _generate_object_state_reasoning(self, pil_frames: list[Image.Image]) -> str:
        """Instruction-agnostic trajectory description (upstream
        ``QwenClient.generate_object_state_reasoning``). Used when
        :attr:`TOPRewardConfig.use_video_description` is ``True``.
        """
        device = next(self.model.parameters()).device
        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames, "fps": self.config.fps},
                    {
                        "type": "text",
                        "text": "Describe the robot manipulation trajectory in this video:",
                    },
                ],
            }
        ]
        prompt_chat = self.processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(user_messages)
        inputs = self.processor(
            text=[prompt_chat],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        self.model.eval()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
        response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        prompt_decoded = self.processor.batch_decode(
            inputs["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        if response.startswith(prompt_decoded):
            return response[len(prompt_decoded) :].strip()
        return response.strip()

    @staticmethod
    def _process_vision_info(messages: list[dict[str, Any]]) -> tuple[Any, Any]:
        """Thin wrapper around ``qwen_vl_utils.process_vision_info``.

        Kept as a method so tests can monkey-patch it without depending on
        the import-time presence of ``qwen_vl_utils``.
        """
        from qwen_vl_utils import process_vision_info

        return cast(tuple[Any, Any], process_vision_info(messages))
