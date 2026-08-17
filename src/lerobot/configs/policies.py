# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import os
import tempfile
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, TypeVar

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError

from lerobot.optim import LRSchedulerConfig, OptimizerConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.device_utils import auto_select_torch_device, is_amp_available, is_torch_device_available
from lerobot.utils.hub import HubMixin

from .types import FeatureType, PolicyFeature

T = TypeVar("T", bound="PreTrainedConfig")
logger = getLogger(__name__)


@dataclass
class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):  # type: ignore[misc,name-defined] #TODO: draccus issue
    """Base configuration class for policy models.

    Every concrete policy config also declares a `normalization_mapping: dict[str, NormalizationMode]`
    field (mapping a `FeatureType` name, e.g. `"STATE"`/`"VISUAL"`, to the `NormalizationMode` to apply),
    with a policy-specific default — not declared here since it has no sensible shared default.

    Args:
        n_obs_steps (`int`, *optional*, defaults to 1): Number of environment steps worth of observations
            to pass to the policy (takes the current step and additional steps going back).
        input_features (`dict[str, PolicyFeature] | None`, *optional*): A dictionary defining the
            `PolicyFeature` of the input data for the policy. The key represents the input data name, and
            the value is a `PolicyFeature`, which consists of `type` and `shape` attributes. Can be set to
            `None`/`null` in order to infer those values from the dataset.
        output_features (`dict[str, PolicyFeature] | None`, *optional*): A dictionary defining the
            `PolicyFeature` of the output data for the policy, with the same key/value semantics as
            `input_features`.
        device (`str | None`, *optional*): The torch device, e.g. `"cuda"`, `"cuda:0"`, `"cpu"`, or `"mps"`.
            If unset or unavailable, `__post_init__` auto-selects one.
        use_amp (`bool`, *optional*, defaults to `False`): Whether to use Automatic Mixed Precision for
            training and evaluation, with automatic gradient scaling. Auto-disabled by `__post_init__`
            when AMP isn't available on `device`.
        use_peft (`bool`, *optional*, defaults to `False`): Whether the policy employed PEFT for training.
        push_to_hub (`bool`, *optional*, defaults to `True`): Whether to push the policy to the Hugging Face
            Hub after training.
        repo_id (`str | None`, *optional*): The Hub repo ID to push to. Required when `push_to_hub` is
            `True`.
        private (`bool | None`, *optional*): Whether to upload to a private repository on the Hugging Face
            Hub.
        tags (`list[str] | None`, *optional*): Tags to add to the policy on the Hub.
        license (`str | None`, *optional*): The license to add to the policy on the Hub.
        pretrained_path (`Path | None`, *optional*): Either the repo ID of a model hosted on the Hub or a
            path to a directory containing weights saved using `PreTrainedPolicy.save_pretrained`. If not
            provided, the policy is initialized from scratch.
        pretrained_revision (`str | None`, *optional*): Hub revision (commit hash, branch, or tag) to pin
            the pretrained model version.
    """

    n_obs_steps: int = 1

    input_features: dict[str, PolicyFeature] | None = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] | None = field(default_factory=dict)

    device: str | None = None
    use_amp: bool = False

    use_peft: bool = False

    push_to_hub: bool = True  # type: ignore[assignment] # TODO: use a different name to avoid override
    repo_id: str | None = None

    private: bool | None = None
    tags: list[str] | None = None
    license: str | None = None
    pretrained_path: Path | None = None
    pretrained_revision: str | None = None

    def __post_init__(self) -> None:
        """Auto-select `device` when unset/unavailable, and disable `use_amp` when AMP isn't available on it."""
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logger.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

        # Automatically deactivate AMP if necessary
        if self.use_amp and not is_amp_available(self.device):
            logger.warning(
                f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
            )
            self.use_amp = False

    @property
    def type(self) -> str:
        """The policy's registered `draccus.ChoiceRegistry` name (e.g. `"act"`, `"diffusion"`)."""
        choice_name = self.get_choice_name(self.__class__)
        if not isinstance(choice_name, str):
            raise TypeError(f"Expected string from get_choice_name, got {type(choice_name)}")
        return choice_name

    @property
    @abc.abstractmethod
    def observation_delta_indices(self) -> list | None:  # type: ignore[type-arg] #TODO: No implementation
        """Offsets, relative to the current step, of the observation timesteps the policy consumes.

        `None` means only the current step is used.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_delta_indices(self) -> list | None:  # type: ignore[type-arg]    #TODO: No implementation
        """Offsets, relative to the current step, of the action timesteps the policy predicts/consumes.

        `None` means only the current step is used.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def reward_delta_indices(self) -> list | None:  # type: ignore[type-arg]    #TODO: No implementation
        """Offsets, relative to the current step, of the reward timesteps the policy consumes.

        `None` means only the current step is used.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        """Return this policy's default `OptimizerConfig`, used when `use_policy_training_preset` is set."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        """Return this policy's default `LRSchedulerConfig`, or `None` if it uses no scheduler."""
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        """Check that `input_features`/`output_features` contain what this policy requires.

        Raises:
            ValueError: If a required feature is missing or has an unexpected shape/type.
        """
        raise NotImplementedError

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        """The input `PolicyFeature` for the robot's proprioceptive state (`observation.state`), if any."""
        if not self.input_features:
            return None
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == OBS_STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        """The input `PolicyFeature` of type `FeatureType.ENV` (environment state), if any."""
        if not self.input_features:
            return None
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        """All input features of type `FeatureType.VISUAL`, keyed by feature name."""
        if not self.input_features:
            return {}
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        """The output `PolicyFeature` for the action (`action`), if any."""
        if not self.output_features:
            return None
        for ft_name, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION and ft_name == ACTION:
                return ft
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        # Encode against the base class so draccus includes the choice "type" key,
        # which `from_pretrained` needs to resolve the concrete subclass.
        with open(save_directory / CONFIG_NAME, "w") as f:
            json.dump(draccus.encode(self, PreTrainedConfig), f, indent=4)

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
        **policy_kwargs: Any,
    ) -> T:
        """Download a policy's `config.json` from the Hub (or read it locally) and parse it.

        The concrete policy config subclass is resolved from the serialized `"type"` tag (e.g. `"act"`,
        `"diffusion"`) rather than being fixed by `cls`, so calling this on the `PreTrainedConfig` base
        class works for any registered policy type.

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
            policy_kwargs: Forwarded as CLI-style overrides via `policy_kwargs["cli_overrides"]`
                (a list of `--key=value` strings applied on top of the loaded config); any other keys are
                ignored.

        Raises:
            FileNotFoundError: If `config.json` isn't found locally or on the Hub.
            ValueError: If `config.json` has no `"type"` field, or its value isn't a registered policy type.
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

        with open(config_file) as f:
            config = json.load(f)

        # Resolve the concrete config subclass from the serialized "type" tag, then parse
        # the config (with CLI overrides) directly for that class. The "type" key is
        # stripped because draccus only consumes it when parsing the registry base class.
        policy_type = config.pop("type", None)
        if policy_type is None:
            raise ValueError(f"Missing 'type' field in {CONFIG_NAME} of {model_id}")
        try:
            config_cls = cls.get_choice_class(policy_type)
        except Exception as e:
            raise ValueError(
                f"Policy type '{policy_type}' (from {CONFIG_NAME} of {model_id}) is not registered. "
                f"Available policy types: {cls.get_known_choices()}"
            ) from e

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            json.dump(config, f)
            config_file = f.name

        cli_overrides = policy_kwargs.pop("cli_overrides", [])
        with draccus.config_type("json"):
            return draccus.parse(config_cls, config_file, args=cli_overrides)
