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

import torch


def to_symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def to_symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


TRANSFORMS = {"symlog": (to_symlog, to_symexp)}
_INV_SQRT2 = 0.7071067811865476


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x * _INV_SQRT2))


class ValueTokenizer:
    """Discretize continuous scalar values into distributional targets."""

    def __init__(
        self,
        bins: int = 256,
        min_value: float = 0.0,
        max_value: float = 1000.0,
        forward_transform=None,
        inverse_transform=None,
        support_transform: str = "linear",
        encoding: str = "two_hot",
        hl_gauss_sigma_ratio: float = 0.75,
        bin_edges=None,
        device=None,
        dtype=torch.float32,
    ) -> None:
        self.n_bins = bins
        self.min_val = float(min_value)
        self.max_val = float(max_value)
        self.support_transform = support_transform
        self.encoding = encoding
        self.hl_gauss_sigma = float(hl_gauss_sigma_ratio)
        self.forward_transform = forward_transform if forward_transform is not None else self.identity
        self.inverse_transform = inverse_transform if inverse_transform is not None else self.identity
        self.device = device
        self.dtype = dtype
        self.is_quantile = support_transform == "quantile"

        if self.is_quantile:
            if bin_edges is None:
                raise ValueError("support_transform='quantile' requires bin_edges of length bins+1.")
            edges = torch.as_tensor(bin_edges, dtype=dtype, device=device)
            if edges.numel() != self.n_bins + 1:
                raise ValueError(
                    f"bin_edges must have length bins+1 ({self.n_bins + 1}), got {edges.numel()}."
                )
            self.edges = edges
            self.center_values = 0.5 * (edges[:-1] + edges[1:])
            self.min_sym = self.max_sym = self.centers_sym = self.bin_stride_sym = None
        else:
            self.edges = self.center_values = None
            self.min_sym = self.forward_transform(torch.tensor(self.min_val, dtype=dtype, device=device))
            self.max_sym = self.forward_transform(torch.tensor(self.max_val, dtype=dtype, device=device))
            self.centers_sym = torch.linspace(
                self.min_sym, self.max_sym, self.n_bins, dtype=dtype, device=device
            )
            self.bin_stride_sym = (
                self.centers_sym[1] - self.centers_sym[0]
                if self.n_bins > 1
                else torch.tensor(1.0, dtype=dtype, device=device)
            )

    @classmethod
    def from_config(cls, config, **kwargs):
        forward_transform = inverse_transform = None
        if config.support_transform in TRANSFORMS:
            forward_transform, inverse_transform = TRANSFORMS[config.support_transform]
        return cls(
            bins=config.bins,
            min_value=config.min_value,
            max_value=config.max_value,
            forward_transform=forward_transform,
            inverse_transform=inverse_transform,
            support_transform=config.support_transform,
            encoding=getattr(config, "encoding", "two_hot"),
            hl_gauss_sigma_ratio=getattr(config, "hl_gauss_sigma_ratio", 0.75),
            bin_edges=getattr(config, "bin_edges", None),
            **kwargs,
        )

    @staticmethod
    def identity(x: torch.Tensor) -> torch.Tensor:
        return x

    def _to_tensor(self, x, device=None):
        if isinstance(x, torch.Tensor):
            return x.to(device=device if device is not None else x.device, dtype=self.dtype)
        return torch.tensor(x, dtype=self.dtype, device=device if device is not None else self.device)

    def _value_to_idx(self, value: torch.Tensor) -> torch.Tensor:
        value = self._to_tensor(value)
        device = value.device
        if self.is_quantile:
            centers = self.center_values.to(device)
            value = torch.clamp(value, min=centers[0], max=centers[-1])
            pos = torch.searchsorted(centers, value, right=True).clamp(1, self.n_bins - 1)
            left, right = centers[pos - 1], centers[pos]
            return (pos - 1).to(value.dtype) + (value - left) / (right - left).clamp_min(1e-12)
        value = torch.clamp(
            value,
            min=torch.tensor(self.min_val, dtype=self.dtype, device=device),
            max=torch.tensor(self.max_val, dtype=self.dtype, device=device),
        )
        return (self.forward_transform(value) - self.min_sym.to(device)) / self.bin_stride_sym.to(device)

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return self.encode_hl_gauss(value) if self.encoding == "hl_gauss" else self.encode_two_hot(value)

    def encode_two_hot(self, value: torch.Tensor) -> torch.Tensor:
        value = self._to_tensor(value)
        idx_float = self._value_to_idx(value)
        idx_left = torch.floor(idx_float).long()
        idx_right = idx_left + 1
        weight_right = idx_float - idx_left.to(idx_float.dtype)
        target = torch.zeros(*value.shape, self.n_bins, dtype=self.dtype, device=value.device)
        target.scatter_add_(
            -1, idx_left.clamp(0, self.n_bins - 1).unsqueeze(-1), (1 - weight_right).unsqueeze(-1)
        )
        target.scatter_add_(-1, idx_right.clamp(0, self.n_bins - 1).unsqueeze(-1), weight_right.unsqueeze(-1))
        return target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def encode_hl_gauss(self, value: torch.Tensor) -> torch.Tensor:
        value = self._to_tensor(value)
        idx_float = self._value_to_idx(value)
        centers = torch.arange(self.n_bins, device=value.device, dtype=idx_float.dtype)
        center = idx_float.unsqueeze(-1)
        sigma = max(self.hl_gauss_sigma, 1e-6)
        probs = _normal_cdf((centers + 0.5 - center) / sigma) - _normal_cdf((centers - 0.5 - center) / sigma)
        return (probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)).to(self.dtype)

    def decode_from_bins(self, bin_logits: torch.Tensor) -> torch.Tensor:
        bin_logits = bin_logits.float()
        if bin_logits.shape[-1] != self.n_bins:
            raise ValueError(
                f"Expected bin_logits last dim == n_bins ({self.n_bins}), got {bin_logits.shape[-1]}"
            )
        probs = torch.softmax(bin_logits, dim=-1)
        if self.is_quantile:
            centers = self.center_values.to(device=bin_logits.device, dtype=bin_logits.dtype)
            return torch.sum(probs * centers, dim=-1)
        centers = self.centers_sym.to(device=bin_logits.device, dtype=bin_logits.dtype)
        return self.inverse_transform(torch.sum(probs * centers, dim=-1))
