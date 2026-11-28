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

"""FlowDAgger noise policy (arXiv 2607.08877).

A small MLP ``pi_w(s)`` that replaces the Gaussian draw at the start of a frozen
flow-matching policy's denoising loop. Its input is the masked mean of the base policy's own
VLM prefix hidden states (images, language and state), so it costs nothing extra to compute
at inference time.

The noise policy is stored separately from the base checkpoint (``config.json`` +
``model.safetensors``) and attached to a pi05 / smolvla policy through the base config's
``noise_policy_path`` field.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

logger = logging.getLogger(__name__)

NOISE_POLICY_CONFIG_NAME = "config.json"
NOISE_POLICY_WEIGHTS_NAME = "model.safetensors"
SUPPORTED_BASE_POLICY_TYPES = ("pi05", "smolvla")
_IN_MEMORY = "<in-memory>"


@dataclass
class NoisePolicyConfig:
    """Architecture of the noise policy and the base policy it was trained against."""

    feature_dim: int
    chunk_size: int
    max_action_dim: int
    base_policy_type: str
    num_inference_steps: int
    hidden_dim: int = 1024
    num_layers: int = 3
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if min(self.feature_dim, self.chunk_size, self.max_action_dim, self.hidden_dim) < 1:
            raise ValueError("feature_dim, chunk_size, max_action_dim and hidden_dim must be positive")

    @property
    def output_dim(self) -> int:
        return self.chunk_size * self.max_action_dim


class NoisePolicy(nn.Module):
    """``standardise -> [Linear -> GELU -> Dropout] x (num_layers - 1) -> Linear`` onto the noise.

    Input features are standardised with per-dimension statistics fitted on the training set
    (:meth:`set_feature_stats`) and stored with the weights. The last layer is
    zero-initialised, so an untrained noise policy outputs the all-zero noise (the mode of the
    base policy's Gaussian prior) rather than an arbitrary sample.
    """

    def __init__(self, config: NoisePolicyConfig):
        super().__init__()
        self.config = config
        self.register_buffer("feature_mean", torch.zeros(config.feature_dim))
        self.register_buffer("feature_std", torch.ones(config.feature_dim))
        layers: list[nn.Module] = []
        in_dim = config.feature_dim
        for _ in range(config.num_layers - 1):
            layers += [nn.Linear(in_dim, config.hidden_dim), nn.GELU(), nn.Dropout(config.dropout)]
            in_dim = config.hidden_dim
        head = nn.Linear(in_dim, config.output_dim)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        layers.append(head)
        self.net = nn.Sequential(*layers)

    @torch.no_grad()
    def set_feature_stats(self, features: Tensor, eps: float = 1e-6) -> None:
        """Fit the input standardisation on ``(N, feature_dim)`` training features."""
        features = features.to(device=self.feature_mean.device, dtype=torch.float32)
        self.feature_mean.copy_(features.mean(dim=0))
        std = features.std(dim=0) if features.shape[0] > 1 else torch.ones_like(self.feature_mean)
        self.feature_std.copy_(std.clamp(min=eps))

    def forward(self, features: Tensor) -> Tensor:
        """Map ``(B, feature_dim)`` features to noise of shape ``(B, chunk_size, max_action_dim)``."""
        x = (features.to(dtype=torch.float32) - self.feature_mean) / self.feature_std
        out = self.net(x)
        return out.view(features.shape[0], self.config.chunk_size, self.config.max_action_dim)

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        (save_directory / NOISE_POLICY_CONFIG_NAME).write_text(json.dumps(asdict(self.config), indent=2))
        state_dict = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(state_dict, str(save_directory / NOISE_POLICY_WEIGHTS_NAME))

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str | Path, **hub_kwargs) -> NoisePolicy:
        """Load from a local directory or a Hugging Face Hub model repo."""
        path = Path(pretrained_name_or_path)
        if path.is_dir():
            config_file = path / NOISE_POLICY_CONFIG_NAME
            weights_file = path / NOISE_POLICY_WEIGHTS_NAME
        else:
            from huggingface_hub import hf_hub_download

            repo_id = str(pretrained_name_or_path)
            config_file = Path(hf_hub_download(repo_id, NOISE_POLICY_CONFIG_NAME, **hub_kwargs))
            weights_file = Path(hf_hub_download(repo_id, NOISE_POLICY_WEIGHTS_NAME, **hub_kwargs))
        config = NoisePolicyConfig(**json.loads(config_file.read_text()))
        policy = cls(config)
        policy.load_state_dict(load_file(str(weights_file)))
        return policy


def check_noise_policy_compatible(
    noise_config: NoisePolicyConfig,
    *,
    policy_type: str,
    chunk_size: int,
    max_action_dim: int,
    feature_dim: int | None = None,
    num_inference_steps: int | None = None,
) -> None:
    """Raise if a noise policy was trained against a differently shaped base policy."""
    mismatches = []
    if noise_config.base_policy_type != policy_type:
        mismatches.append(f"policy type {noise_config.base_policy_type!r} != {policy_type!r}")
    if noise_config.chunk_size != chunk_size:
        mismatches.append(f"chunk_size {noise_config.chunk_size} != {chunk_size}")
    if noise_config.max_action_dim != max_action_dim:
        mismatches.append(f"max_action_dim {noise_config.max_action_dim} != {max_action_dim}")
    if feature_dim is not None and noise_config.feature_dim != feature_dim:
        mismatches.append(f"feature_dim {noise_config.feature_dim} != {feature_dim}")
    if mismatches:
        raise ValueError("Noise policy is incompatible with the base policy: " + "; ".join(mismatches))
    if num_inference_steps is not None and noise_config.num_inference_steps != num_inference_steps:
        # Not fatal: the inverted noise was computed for a different integrator, so it is
        # only approximately right, but the user may want to experiment with this.
        logger.warning(
            "Noise policy was trained with %d denoising steps but the base policy uses %d; "
            "the adapted actions will not match the inverted targets exactly.",
            noise_config.num_inference_steps,
            num_inference_steps,
        )


def masked_mean_pool(hidden_states: Tensor, pad_mask: Tensor) -> Tensor:
    """Mean of ``(B, L, D)`` hidden states over the positions where ``pad_mask`` is True, in fp32."""
    mask = pad_mask.to(device=hidden_states.device, dtype=torch.float32).unsqueeze(-1)
    summed = (hidden_states.to(torch.float32) * mask).sum(dim=1)
    return summed / mask.sum(dim=1).clamp(min=1.0)


class NoisePolicyHolder:
    """Lazily loads the noise policy named by ``config.noise_policy_path`` for a base policy.

    The noise policy is held outside the ``nn.Module`` tree on purpose, so that saving,
    pushing or ``state_dict()``-ing the base policy never includes it.
    """

    def __init__(self) -> None:
        self._noise_policy: NoisePolicy | None = None
        self._loaded_from: str | None = None

    def get(
        self,
        path: str | None,
        *,
        policy_type: str,
        chunk_size: int,
        max_action_dim: int,
        num_inference_steps: int,
        device: torch.device,
    ) -> NoisePolicy | None:
        if self._loaded_from == _IN_MEMORY:
            pass  # an explicitly attached noise policy wins over the config path
        elif not path:
            self._noise_policy = None
            self._loaded_from = None
            return None
        elif self._noise_policy is None or self._loaded_from != path:
            noise_policy = NoisePolicy.from_pretrained(path)
            check_noise_policy_compatible(
                noise_policy.config,
                policy_type=policy_type,
                chunk_size=chunk_size,
                max_action_dim=max_action_dim,
                num_inference_steps=num_inference_steps,
            )
            noise_policy.eval().requires_grad_(False)
            self._noise_policy = noise_policy
            self._loaded_from = path
            logger.info("FlowDAgger: loaded noise policy from %s", path)
        assert self._noise_policy is not None
        if next(self._noise_policy.parameters()).device != device:
            self._noise_policy.to(device)
        return self._noise_policy

    def set(self, noise_policy: NoisePolicy | None) -> None:
        """Attach an in-memory noise policy (e.g. right after training).

        The holder lives outside the base policy's module tree, so ``policy.eval()`` never
        reaches the noise policy: freeze it here, as for path-loaded noise policies, so that
        dropout cannot make the starting noise non-deterministic.
        """
        if noise_policy is not None:
            noise_policy.eval().requires_grad_(False)
        self._noise_policy = noise_policy
        self._loaded_from = _IN_MEMORY if noise_policy is not None else None
