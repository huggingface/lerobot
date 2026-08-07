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
        input_features (`dict[str, PolicyFeature]`, *optional*): A dictionary defining the `PolicyFeature`
            of the input data for the reward. The key represents the input data name, and the value is a
            `PolicyFeature`, which consists of `type` and `shape` attributes.
        output_features (`dict[str, PolicyFeature]`, *optional*): A dictionary defining the `PolicyFeature`
            of the output data for the reward, with the same key/value semantics as `input_features`.
        device (`str | None`, *optional*): The torch device, e.g. `"cuda"`, `"cuda:0"`, `"cpu"`, or `"mps"`.
            If unset or unavailable, `__post_init__` auto-selects one.
        pretrained_path (`str | None`, *optional*): Either the repo ID of a model hosted on the Hub or a
            path to a directory containing weights saved using `.save_pretrained`. If not provided, the
            reward model is initialized from scratch.
        pretrained_revision (`str | None`, *optional*): Optional Hub revision, e.g. a commit hash, branch,
            or tag, to pin the pretrained reward model version.
        push_to_hub (`bool`, *optional*, defaults to `False`): Whether to push the reward model to the
            Hugging Face Hub after training.
        repo_id (`str | None`, *optional*): The Hub repo ID to push to. Required when `push_to_hub` is
            `True`.
        license (`str | None`, *optional*): The license to add to the reward model on the Hub.
        tags (`list[str] | None`, *optional*): Tags to add to the reward model on the Hub.
        private (`bool | None`, *optional*): Whether to upload to a private repository on the Hugging Face
            Hub.
    """

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    device: str | None = None

    pretrained_path: str | None = None
    pretrained_revision: str | None = None

    push_to_hub: bool = False
    repo_id: str | None = None

    license: str | None = None
    tags: list[str] | None = None
    private: bool | None = None

    def __post_init__(self) -> None:
        """Auto-select `device` when unset or unavailable."""
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logger.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

    @property
    def type(self) -> str:
        """The reward model's registered `draccus.ChoiceRegistry` name."""
        choice_name = self.get_choice_name(self.__class__)
        if not isinstance(choice_name, str):
            raise TypeError(f"Expected string from get_choice_name, got {type(choice_name)}")
        return choice_name

    @property
    def observation_delta_indices(self) -> list | None:  # type: ignore[type-arg]
        """`None`: reward models consume only the current observation timestep."""
        return None

    @property
    def action_delta_indices(self) -> list | None:  # type: ignore[type-arg]
        """`None`: reward models consume only the current action timestep."""
        return None

    @property
    def reward_delta_indices(self) -> list | None:  # type: ignore[type-arg]
        """`None`: reward models consume only the current reward timestep."""
        return None

    def get_optimizer_preset(self) -> OptimizerConfig | None:
        """Default optimizer for this reward model, or ``None`` for zero-shot models."""
        return None

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        """Default LR scheduler for this reward model. `None` here; overridden by subclasses that need one."""
        return None

    def validate_features(self) -> None:
        """Check that `input_features`/`output_features` contain what this reward model requires.

        No-op here; overridden by subclasses that have required features.
        """
        pass

    def _save_pretrained(self, save_directory: Path) -> None:
        # Encode against the base class so draccus includes the choice "type" key,
        # which `from_pretrained` needs to resolve the concrete subclass.
        with open(save_directory / CONFIG_NAME, "w") as f:
            json.dump(draccus.encode(self, RewardModelConfig), f, indent=4)

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
        """Download a reward model's `config.json` from the Hub (or read it locally) and parse it.

        The concrete reward-model config subclass is resolved from the serialized `"type"` tag, so
        calling this on the `RewardModelConfig` base class works for any registered reward-model type.

        Args:
            pretrained_name_or_path (`str | Path`): Either the `repo_id` of the config hosted on the Hub,
                or a path to a directory containing a `config.json` saved via `.save_pretrained`.
            force_download (`bool`, *optional*, defaults to `False`): Whether to force (re-)downloading
                the files from the Hub, overriding the existing cache.
            resume_download (`bool | None`, *optional*): Deprecated; ignored by the underlying Hub client.
            proxies (`dict[Any, Any] | None`, *optional*): A dictionary of proxy servers to use by protocol
                or endpoint.
            token (`str | bool | None`, *optional*): The token to use as HTTP bearer authorization for
                remote files. By default, uses the token cached by `huggingface-cli login`.
            cache_dir (`str | Path | None`, *optional*): Path to the folder where cached files are stored.
            local_files_only (`bool`, *optional*, defaults to `False`): If `True`, avoid downloading the
                file and return the path to the local cached file if it exists.
            revision (`str | None`, *optional*): Revision on the Hub: a branch name, git tag, or commit id.
                Defaults to the latest commit on `main`.
            reward_kwargs: Forwarded as CLI-style overrides via `reward_kwargs["cli_overrides"]`
                (a list of `--key=value` strings applied on top of the loaded config); any other keys are
                ignored.

        Raises:
            FileNotFoundError: If `config.json` isn't found locally or on the Hub.
        """
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
