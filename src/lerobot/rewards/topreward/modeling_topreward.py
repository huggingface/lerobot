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

1. The processor builds a chat-style prompt, tokenises it, and emits
   ``input_ids``, ``attention_mask``, vision tensors, and ``prompt_length``.
2. The model label-masks everything before ``prompt_length`` with ``-100``.
3. Forward the full token sequence through the VLM.
4. Read per-token log-probabilities of the unmasked suffix tokens from the
   logits and reduce them (mean or sum) into a scalar reward.

With the default ``prompt_suffix_template`` and ``prompt_length = input_len - 1``
(mirrored from upstream), the only unmasked token is the literal ``"True"``
at the end — the reward is ``log P("True" | video + prompt + instruction)``.

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
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError
from torch import Tensor

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.rewards.topreward.configuration_topreward import TOPRewardConfig
from lerobot.rewards.topreward.processor_topreward import TOPREWARD_FEATURE_PREFIX, TOPREWARD_INPUT_KEYS
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig

if TYPE_CHECKING or _transformers_available:
    from transformers import Qwen3VLForConditionalGeneration
else:
    Qwen3VLForConditionalGeneration = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="TOPRewardModel")


def _torch_dtype(name: str) -> torch.dtype | str:
    """Resolve a torch dtype name; ``"auto"`` is passed through verbatim."""
    if name == "auto":
        return "auto"
    dtype = getattr(torch, name, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    raise ValueError(f"Unknown torch dtype: {name!r}")


class TOPRewardModel(PreTrainedRewardModel):
    """TOPReward zero-shot reward model."""

    name = "topreward"
    config_class = TOPRewardConfig

    def __init__(self, config: TOPRewardConfig) -> None:
        require_package("transformers", extra="topreward")
        super().__init__(config)
        self.config = config

        torch_dtype = _torch_dtype(config.torch_dtype)
        model_kwargs: dict[str, Any] = {"dtype": torch_dtype, "trust_remote_code": True}
        if config.attn_implementation is not None:
            model_kwargs["attn_implementation"] = config.attn_implementation

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(config.vlm_name, **model_kwargs)

    def compute_reward(self, batch: dict[str, Any]) -> Tensor:
        """Return one log-prob reward per sample in the batch."""
        inputs = {
            key: batch[f"{TOPREWARD_FEATURE_PREFIX}{key}"]
            for key in TOPREWARD_INPUT_KEYS
            if f"{TOPREWARD_FEATURE_PREFIX}{key}" in batch
        }
        if "input_ids" not in inputs:
            raise KeyError(
                f"TOPReward batch missing pre-encoded inputs (expected "
                f"`{TOPREWARD_FEATURE_PREFIX}input_ids`). Make sure the "
                "TOPRewardEncoderProcessorStep ran before `compute_reward`."
            )

        prompt_lengths = inputs.pop("prompt_length")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

        labels = inputs["input_ids"].clone()
        for i, plen in enumerate(prompt_lengths.tolist()):
            labels[i, : int(plen)] = -100
        if "attention_mask" in inputs:
            labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

        self.eval()
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)

        logits = outputs.logits[:, :-1, :]
        target_labels = labels[:, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        mask = target_labels != -100
        safe_targets = target_labels.masked_fill(~mask, 0)
        token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)

        batch_size = inputs["input_ids"].shape[0]
        rewards = []
        for i in range(batch_size):
            sample_log_probs = token_log_probs[i][mask[i]]
            if sample_log_probs.numel() == 0:
                raise RuntimeError(
                    "TOPReward could not isolate any suffix tokens to score. Check that "
                    "`prompt_suffix_template` produces at least one tokenised character."
                )
            if self.config.reduction == "sum":
                rewards.append(sample_log_probs.sum().item())
            else:
                rewards.append(sample_log_probs.mean().item())

        out = torch.as_tensor(rewards, dtype=torch.float32)
        if np.isfinite(self.config.success_threshold):
            out = (out > self.config.success_threshold).float()
        return out.to(self.config.device or "cpu")

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save ``config.json`` only."""
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
        strict: bool = False,  # noqa: ARG003 — accepted for API parity; unused (no safetensors to load)
        **kwargs: Any,
    ) -> T:
        """Load a TOPReward configuration and instantiate the wrapped VLM."""
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
        """Push the TOPReward ``config.json`` + model card to the Hub."""
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
