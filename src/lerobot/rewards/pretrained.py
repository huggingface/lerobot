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

import abc
import builtins
import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_model as load_model_as_safetensor, save_model as save_model_as_safetensor
from torch import Tensor, nn

from lerobot.configs.rewards import RewardModelConfig
from lerobot.utils.device_utils import resolve_safetensors_device
from lerobot.utils.hub import HubMixin

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig

T = TypeVar("T", bound="PreTrainedRewardModel")


class PreTrainedRewardModel(nn.Module, HubMixin, abc.ABC):
    """Base class for reward models."""

    config_class: None
    name: None

    def __init__(self, config: RewardModelConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, RewardModelConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`RewardModelConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "config_class", None):
            raise TypeError(f"Class {cls.__name__} must define 'config_class'")
        if not getattr(cls, "name", None):
            raise TypeError(f"Class {cls.__name__} must define 'name'")

    def _save_pretrained(self, save_directory: Path) -> None:
        """Serialize this reward model's parameters (and config) into `save_directory`.

        Safe to call on every rank: replicas carry identical weights, so only the main process
        writes (sharded reward models are rejected at config validation — no collective gather).

        Args:
            save_directory (Path): Target directory for the reward model config (`config.json`)
                and `model.safetensors`.
        """
        from lerobot.distributed.utils import is_main_process

        # save_checkpoint calls this on every rank; replicas carry identical
        # weights, so the main process is the only writer. Sharded reward models are rejected
        # at config validation, so no collective gather is needed here.
        if not is_main_process():
            return
        self.config._save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))

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
        strict: bool = False,
        **kwargs,
    ) -> T:
        """
        The reward model is set in evaluation mode by default using `reward.eval()` (dropout modules are
        deactivated). To train it, you should first set it back in training mode with `reward.train()`.
        """
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
        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            reward = cls._load_as_safetensor(instance, model_file, config.device or "cpu", strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                reward = cls._load_as_safetensor(instance, model_file, config.device or "cpu", strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        reward.to(config.device)
        reward.eval()
        return reward

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        missing_keys, unexpected_keys = load_model_as_safetensor(
            model, model_file, strict=strict, device=resolve_safetensors_device(map_location)
        )
        if missing_keys:
            logging.warning(f"Missing key(s) when loading model: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected key(s) when loading model: {unexpected_keys}")
        return model

    def get_optim_params(self):
        """
        Returns the reward-model-specific parameters dict to be passed on to the optimizer.
        """
        return self.parameters()

    def reset(self) -> None:
        """Reset any internal state."""
        pass

    @abc.abstractmethod
    def compute_reward(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute a scalar reward signal for a batch of observations.

        Args:
            batch: Dictionary containing at minimum observation tensors.
                   May also contain "action", "next_observation.*", etc.

        Returns:
            Tensor of shape ``(batch_size,)`` with reward values.
        """
        ...

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
        """Training forward pass — override for trainable reward models."""
        raise NotImplementedError(
            f"{self.__class__.__name__} is not trainable. Only use compute_reward() for inference."
        )

    @property
    def is_trainable(self) -> bool:
        """Whether this reward model can be trained via ``lerobot-train``.

        Trainable reward models override :meth:`forward`; zero-shot models
        inherit the base implementation that raises ``NotImplementedError``.
        """
        return type(self).forward is not PreTrainedRewardModel.forward

    def push_model_to_hub(self, cfg: "TrainPipelineConfig") -> None:
        """Publish this reward model to the Hub.

        Deprecated: use :func:`lerobot.common.train_utils.publish_trained_model` instead.

        Args:
            cfg (TrainPipelineConfig): The training config; saved as `train_config.json` and
                used to render the model card.
        """
        from lerobot.common.train_utils import publish_trained_model

        warnings.warn(
            "PreTrainedRewardModel.push_model_to_hub is deprecated and will be removed in a "
            "future version. Use lerobot.common.train_utils.publish_trained_model(cfg, model, "
            "preprocessor, postprocessor, dataset_meta) instead.",
            FutureWarning,
            stacklevel=2,
        )
        publish_trained_model(cfg, self, None, None, None)
