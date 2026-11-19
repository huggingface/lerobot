# Copyright 2026 Alibaba DAMO Academy and The HuggingFace Inc. team. All rights reserved.
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

# value_tokenizer.py

import torch


def to_symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def to_symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


TRANSFORMS = {"symlog": (to_symlog, to_symexp)}


_INV_SQRT2 = 0.7071067811865476


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF Phi(x) via erf."""
    return 0.5 * (1.0 + torch.erf(x * _INV_SQRT2))


class ValueTokenizer:
    """
    Tokenizer for continuous scalar values using bin discretization.

    Two orthogonal choices control the behaviour:

    * ``support_transform`` — how a scalar maps to a fractional bin index:
        - ``"linear"`` / ``"symlog"``: uniform bins in (transformed) support
          space, ``idx = (transform(y) - min_sym) / stride`` over ``n_bins``
          bin centers.
        - ``"quantile"``: non-uniform bins whose centers are the midpoints of
          data-driven ``bin_edges`` (empirical CDF). The scalar is mapped to a
          fractional center index by piecewise-linear interpolation.
    * ``encoding`` — how the fractional index becomes a target distribution:
        - ``"two_hot"``: linear split over the two adjacent bin centers.
        - ``"hl_gauss"``: Gaussian centered at the fractional index, integrated
          over each bin's unit interval in center-index space (HL-Gauss,
          Farebrother et al. 2024), then renormalized (tails absorbed).

    All encodings share one internal "center-index space": ``n_bins`` centers at
    integer positions ``0 .. n_bins-1``. This keeps ``two_hot`` numerically
    identical to the legacy implementation for the linear/symlog modes.
    """

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
        # sigma expressed in center-index (== bin width) units.
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
            # Bin centers in value space: midpoint of each [edge_i, edge_{i+1}].
            self.center_values = 0.5 * (edges[:-1] + edges[1:])
            # Uniform-space attributes are unused in quantile mode.
            self.min_sym = None
            self.max_sym = None
            self.centers_sym = None
            self.bin_stride_sym = None
        else:
            self.edges = None
            self.center_values = None
            self.min_sym = self.forward_transform(torch.tensor(self.min_val, dtype=dtype, device=device))
            self.max_sym = self.forward_transform(torch.tensor(self.max_val, dtype=dtype, device=device))
            self.centers_sym = torch.linspace(
                self.min_sym, self.max_sym, self.n_bins, dtype=dtype, device=device
            )
            if self.n_bins > 1:
                self.bin_stride_sym = self.centers_sym[1] - self.centers_sym[0]
            else:
                self.bin_stride_sym = torch.tensor(1.0, dtype=dtype, device=device)

    @classmethod
    def from_config(cls, config, **kwargs):
        forward_transform = None
        inverse_transform = None
        support_transform = config.support_transform
        if support_transform in TRANSFORMS:
            forward_transform, inverse_transform = TRANSFORMS[support_transform]

        return cls(
            bins=config.bins,
            min_value=config.min_value,
            max_value=config.max_value,
            forward_transform=forward_transform,
            inverse_transform=inverse_transform,
            support_transform=support_transform,
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
        return torch.tensor(
            x,
            dtype=self.dtype,
            device=device if device is not None else self.device,
        )

    def _value_to_idx(self, value: torch.Tensor) -> torch.Tensor:
        """Map scalar values to a fractional center index in ``[0, n_bins-1]``.

        Args:
            value: Tensor with shape (...,)

        Returns:
            Tensor with shape (...,), the fractional bin-center index.
        """
        value = self._to_tensor(value)
        device = value.device

        if self.is_quantile:
            centers = self.center_values.to(device)
            v = torch.clamp(value, min=centers[0], max=centers[-1])
            # First center strictly greater than v (after equal elements).
            pos = torch.searchsorted(centers, v, right=True)
            pos = pos.clamp(1, self.n_bins - 1)
            c_left = centers[pos - 1]
            c_right = centers[pos]
            frac = (v - c_left) / (c_right - c_left).clamp_min(1e-12)
            return (pos - 1).to(v.dtype) + frac

        min_val = torch.tensor(self.min_val, dtype=self.dtype, device=device)
        max_val = torch.tensor(self.max_val, dtype=self.dtype, device=device)
        min_sym = self.min_sym.to(device)
        bin_stride_sym = self.bin_stride_sym.to(device)

        value = torch.clamp(value, min=min_val, max=max_val)
        v_sym = self.forward_transform(value)
        return (v_sym - min_sym) / bin_stride_sym

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        """Encode scalars into a target distribution over bins (dispatch)."""
        if self.encoding == "hl_gauss":
            return self.encode_hl_gauss(value)
        return self.encode_two_hot(value)

    def encode_two_hot(self, value: torch.Tensor) -> torch.Tensor:
        """
        Convert scalar values into a two-hot distribution over bins.

        Args:
            value: Tensor with shape (...,)

        Returns:
            Tensor with shape (..., n_bins)
        """
        value = self._to_tensor(value)
        device = value.device

        idx_float = self._value_to_idx(value)
        idx_left = torch.floor(idx_float).long()
        idx_right = idx_left + 1

        weight_right = idx_float - idx_left.to(idx_float.dtype)
        weight_left = 1.0 - weight_right

        idx_left_clamped = idx_left.clamp(0, self.n_bins - 1)
        idx_right_clamped = idx_right.clamp(0, self.n_bins - 1)

        target_dist = torch.zeros(*value.shape, self.n_bins, dtype=self.dtype, device=device)

        target_dist.scatter_add_(
            dim=-1,
            index=idx_left_clamped.unsqueeze(-1),
            src=weight_left.unsqueeze(-1),
        )
        target_dist.scatter_add_(
            dim=-1,
            index=idx_right_clamped.unsqueeze(-1),
            src=weight_right.unsqueeze(-1),
        )

        target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return target_dist

    def encode_hl_gauss(self, value: torch.Tensor) -> torch.Tensor:
        """
        Convert scalar values into an HL-Gauss distribution over bins.

        A Gaussian centered at the fractional bin-center index (std
        ``hl_gauss_sigma`` in center-index units) is integrated over each bin's
        unit interval ``[i-0.5, i+0.5]``; the tails outside the support are
        absorbed by renormalization.

        Args:
            value: Tensor with shape (...,)

        Returns:
            Tensor with shape (..., n_bins)
        """
        value = self._to_tensor(value)
        device = value.device

        idx_float = self._value_to_idx(value)
        sigma = max(self.hl_gauss_sigma, 1e-6)

        centers = torch.arange(self.n_bins, device=device, dtype=idx_float.dtype)
        c = idx_float.unsqueeze(-1)
        cdf_upper = _normal_cdf((centers + 0.5 - c) / sigma)
        cdf_lower = _normal_cdf((centers - 0.5 - c) / sigma)
        probs = cdf_upper - cdf_lower

        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return probs.to(self.dtype)

    def decode_from_bins(self, bin_logits: torch.Tensor) -> torch.Tensor:
        """
        Decode scalar predictions from bin logits.

        Args:
            bin_logits: Tensor with shape (..., n_bins)

        Returns:
            Tensor with shape (...,)
        """
        bin_logits = bin_logits.float()
        if bin_logits.shape[-1] != self.n_bins:
            raise ValueError(
                f"Expected bin_logits last dim == n_bins ({self.n_bins}), got {bin_logits.shape[-1]}"
            )

        probs = torch.softmax(bin_logits, dim=-1)

        if self.is_quantile:
            centers = self.center_values.to(device=bin_logits.device, dtype=bin_logits.dtype)
            return torch.sum(probs * centers, dim=-1)

        centers_sym = self.centers_sym.to(device=bin_logits.device, dtype=bin_logits.dtype)
        pred_value_sym = torch.sum(probs * centers_sym, dim=-1)
        return self.inverse_transform(pred_value_sym)
