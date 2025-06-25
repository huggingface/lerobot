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
import logging
import os
from importlib.resources import files
from pathlib import Path
from typing import Type, TypeVar

import packaging
import safetensors
from huggingface_hub import ModelCard, ModelCardData, hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_model as load_model_as_safetensor
from safetensors.torch import save_model as save_model_as_safetensor
from torch import Tensor, nn

from lerobot.common.utils.hub import HubMixin
from lerobot.configs.policies import PreTrainedConfig

T = TypeVar("T", bound="PreTrainedPolicy")


class PreTrainedPolicy(nn.Module, HubMixin, abc.ABC):
    """
    Base class for policy models.
    """

    config_class: None
    name: None

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
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
        self.config._save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))

        card = self.generate_model_card()
        card.save(str(save_directory / "README.md"))

    @classmethod
    def from_pretrained(
        cls: Type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
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
        The policy is set in evaluation mode by default using `policy.eval()` (dropout modules are
        deactivated). To train it, you should first set it back in training mode with `policy.train()`.
        """
        if config is None:
            config = PreTrainedConfig.from_pretrained(
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
            policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
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
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        policy.to(config.device)
        policy.eval()
        return policy

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        if packaging.version.parse(safetensors.__version__) < packaging.version.parse("0.4.3"):
            load_model_as_safetensor(model, model_file, strict=strict)
            if map_location != "cpu":
                logging.warning(
                    "Loading model weights on other devices than 'cpu' is not supported natively in your version of safetensors."
                    " This means that the model is loaded on 'cpu' first and then copied to the device."
                    " This leads to a slower loading time."
                    " Please update safetensors to version 0.4.3 or above for improved performance."
                )
                model.to(map_location)
        else:
            safetensors.torch.load_model(model, model_file, strict=strict, device=map_location)
        return model

    def generate_model_card(self) -> ModelCard:
        dataset_repo_id = self.config.dataset_repo_id
        datasets = [dataset_repo_id] if dataset_repo_id and isinstance(dataset_repo_id, str) else None

        model_name = self.name  # This is the policy name
        base_model = "lerobot/smolvla_base" if model_name == "smolvla" else None  # Set a base model
        model_summary = ""

        if model_name == "smolvla":
            model_summary = "[SmolVLA](https://huggingface.co/papers/2506.01844) is a compact, efficient vision-language-action model that achieves competitive performance at reduced computational costs and can be deployed on consumer-grade hardware."
        elif model_name == "act":
            model_summary = "[Action Chunking with Transformers (ACT)](https://huggingface.co/papers/2304.13705) is a imitation-learning method that, predicts short action chunks instead of single steps. It learns from tele-operated data and often achieves high success rates."
        elif model_name == "tdmpc":
            model_summary = "[TD-MPC](https://huggingface.co/papers/2203.04955) combines model-free and model-based approaches to improve sample efficiency and performance in continuous control tasks by using a learned latent dynamics model and terminal value function."
        elif model_name == "diffusion":
            model_summary = "[Diffusion Policy](https://huggingface.co/papers/2303.04137) treats visuomotor control as a generative diffusion process, producing smooth, multi-step action trajectories that excel at contact-rich manipulation."
        elif model_name == "vqbet":
            model_summary = "[VQ-BET](https://huggingface.co/papers/2403.03181) combines vector-quantised action tokens with Behaviour Transformers to discretise control and achieve data-efficient imitation across diverse skills."
        elif model_name == "pi0":
            model_summary = "[Pi0](https://huggingface.co/papers/2410.24164) is a generalist vision-language-action transformer that converts multimodal observations and text instructions into robot actions for zero-shot task transfer."
        elif model_name == "pi0fast":
            model_summary = "[Pi0-Fast](https://huggingface.co/papers/2501.09747) is variant of Pi0 that uses a new tokenization method called FAST, which enables training of a autoregressive vision-language-action policy for high-frequency robotic tasks with improved performance and reduced training time."
        elif model_name == "sac":
            model_summary = "[Soft Actor-Critic (SAC)](https://huggingface.co/papers/1801.01290) is an entropy-regularised actor-critic algorithm offering stable, sample-efficient learning in continuous-control environments."
        elif model_name == "reward_classifier":
            model_summary = "A reward classifier is a lightweight neural network that scores observations or trajectories for task success, providing a learned reward signal or offline evaluation when explicit rewards are unavailable."

        card_data = ModelCardData(
            license="apache-2.0",
            library_name="lerobot",
            pipeline_tag="robotics",
            tags=["robotics", model_name],
            model_name=model_name,
            datasets=datasets,
            base_model=base_model,
            model_summary=model_summary,
        )

        template_path = files("lerobot.templates").joinpath("lerobot_modelcard_template.md")
        card = ModelCard.from_template(card_data, template_path=str(template_path))
        card.validate()
        return card

    @abc.abstractmethod
    def get_optim_params(self) -> dict:
        """
        Returns the policy-specific parameters dict to be passed on to the optimizer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """To be called whenever the environment is reset.

        Does things like clearing caches.
        """
        raise NotImplementedError

    # TODO(aliberts, rcadene): split into 'forward' and 'compute_loss'?
    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """_summary_

        Args:
            batch (dict[str, Tensor]): _description_

        Returns:
            tuple[Tensor, dict | None]: The loss and potentially other information. Apart from the loss which
                is a Tensor, all other items should be logging-friendly, native Python types.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Return one action to run in the environment (potentially in batch mode).

        When the model uses a history of observations, or outputs a sequence of actions, this method deals
        with caching.
        """
        raise NotImplementedError
