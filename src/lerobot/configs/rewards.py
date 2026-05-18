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
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError

from lerobot.optim.optimizers import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.device_utils import auto_select_torch_device, is_torch_device_available
from lerobot.utils.hub import HubMixin

from .types import PolicyFeature

T = TypeVar("T", bound="RewardModelConfig")
logger = logging.getLogger(__name__)


@dataclass
class RewardModelConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):
    """Base configuration for reward models.

    Args:
    input_features: A dictionary defining the PolicyFeature of the input data for the reward. The key represents
        the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
    output_features: A dictionary defining the PolicyFeature of the output data for the reward. The key represents
        the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
    """

    # Reuses PolicyFeature
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    device: str | None = None

    pretrained_path: str | None = None

    push_to_hub: bool = False
    repo_id: str | None = None

    # Hub metadata
    license: str | None = None
    tags: list[str] | None = None
    private: bool | None = None

    def __post_init__(self) -> None:
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logger.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

    @property
    def type(self) -> str:
        choice_name = self.get_choice_name(self.__class__)
        if not isinstance(choice_name, str):
            raise TypeError(f"Expected string from get_choice_name, got {type(choice_name)}")
        return choice_name

    @property
    def observation_delta_indices(self) -> list | None:  # type: ignore[type-arg]
        return None

    @property
    def action_delta_indices(self) -> list | None:  # type: ignore[type-arg]
        return None

    @property
    def reward_delta_indices(self) -> list | None:  # type: ignore[type-arg]
        return None

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        pass

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **reward_kwargs: Any,
    ) -> T:
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                logger.error(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
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

        if config_file is None:
            raise FileNotFoundError(f"{CONFIG_NAME} not found in {model_id}")

        # HACK: Parse the original config to get the config subclass, so that we can
        # apply cli overrides.
        with draccus.config_type("json"):
            orig_config = draccus.parse(cls, config_file, args=[])

        with open(config_file) as f:
            config = json.load(f)

        config.pop("type", None)
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            json.dump(config, f)
            config_file = f.name

        cli_overrides = reward_kwargs.pop("cli_overrides", [])
        with draccus.config_type("json"):
            return draccus.parse(orig_config.__class__, config_file, args=cli_overrides)
