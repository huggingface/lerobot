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

import abc
import builtins
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
from torch.optim import Optimizer

from lerobot.types import BatchType
from lerobot.utils.hub import HubMixin

from .configs import RLAlgorithmConfig, TrainingStats

if TYPE_CHECKING:
    from torch import nn

    from ..data_sources.data_mixer import DataMixer

T = TypeVar("T", bound="RLAlgorithm")


class RLAlgorithm(HubMixin, abc.ABC):
    """Base for all RL algorithms."""

    config_class: type[RLAlgorithmConfig]
    name: str
    config: RLAlgorithmConfig

    @abc.abstractmethod
    def update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        """One complete training step.

        The algorithm calls ``next(batch_iterator)`` as many times as it
        needs (e.g. ``utd_ratio`` times for SAC) to obtain fresh batches.
        The iterator is owned by the trainer; the algorithm just consumes
        from it.
        """
        raise NotImplementedError

    def configure_data_iterator(
        self,
        data_mixer: DataMixer,
        batch_size: int,
        *,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ) -> Iterator[BatchType]:
        """Create the data iterator this algorithm needs.

        The default implementation uses the standard ``data_mixer.get_iterator()``.
        Algorithms that need specialised sampling should override this method.
        """
        return data_mixer.get_iterator(
            batch_size=batch_size,
            async_prefetch=async_prefetch,
            queue_size=queue_size,
        )

    @abc.abstractmethod
    def make_optimizers_and_scheduler(self) -> dict[str, Optimizer]:
        """Build and return the optimizers used during training.

        Called once on the learner side after construction.
        """
        raise NotImplementedError

    def get_optimizers(self) -> dict[str, Optimizer]:
        """Return optimizers for checkpointing / external scheduling."""
        return {}

    @property
    def optimization_step(self) -> int:
        """Current learner optimization step.

        Part of the stable contract for checkpoint/resume. Algorithms can
        either use this default storage or override for custom behavior.
        """
        return getattr(self, "_optimization_step", 0)

    @optimization_step.setter
    def optimization_step(self, value: int) -> None:
        self._optimization_step = int(value)

    def get_weights(self) -> dict[str, Any]:
        """Policy state-dict to push to actors."""
        return {}

    @abc.abstractmethod
    def load_weights(self, weights: dict[str, Any], device: str | torch.device = "cpu") -> None:
        """Load policy state-dict received from the learner."""
        raise NotImplementedError

    @abc.abstractmethod
    def state_dict(self) -> dict[str, torch.Tensor]:
        """Algorithm-owned trainable tensors.

        Must return a flat tensor mapping for everything the algorithm owns
        that is not part of the policy (e.g. critic ensembles, target networks,
        temperature parameters). Algorithms with no training-only tensors
        should explicitly return an empty dict.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device = "cpu",
    ) -> None:
        """In-place load of algorithm-owned tensors.

        Implementations MUST keep the identity of any ``nn.Parameter`` that an
        optimizer references (e.g. SAC's ``log_alpha``) by using ``.copy_()``
        rather than rebinding the attribute.
        """
        raise NotImplementedError

    def _save_pretrained(self, save_directory: Path) -> None:
        """Persist the algorithm's tensors and config to ``save_directory``.

        Writes ``model.safetensors`` (algorithm tensors via :meth:`state_dict`)
        and ``config.json`` (via :meth:`RLAlgorithmConfig.save_pretrained`).
        """
        tensors = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_safetensors(tensors, str(save_directory / SAFETENSORS_SINGLE_FILE))
        self.config._save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        policy: nn.Module,
        config: RLAlgorithmConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        device: str | torch.device = "cpu",
        **algo_kwargs: Any,
    ) -> T:
        """Build an algorithm and load its weights from ``pretrained_name_or_path``."""
        if config is None:
            config = cls.config_class.from_pretrained(
                pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
            )
        if hasattr(config, "policy_config"):
            config.policy_config = policy.config

        instance = cls(policy=policy, config=config, **algo_kwargs)

        model_id = str(pretrained_name_or_path)
        if os.path.isdir(model_id):
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
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
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        tensors = load_safetensors(model_file)
        instance.load_state_dict(tensors, device=device)
        return instance
