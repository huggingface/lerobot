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
# Note: We subclass str so that serialization is straightforward
# https://stackoverflow.com/questions/24481852/serialising-an-enum-member-to-json
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

import draccus
import torch


def _coerce_dtype(value: str | torch.dtype | None) -> torch.dtype | None:
    """Normalize a dtype request to a floating ``torch.dtype``, or ``None`` for "unspecified".

    Accepts a ``torch.dtype``, the name of one (``"bfloat16"``, or ``"torch.bfloat16"``), or
    ``None``. Any floating-point dtype PyTorch exposes is valid -- there is deliberately no
    allowlist, so ``float64`` and the ``float8_*`` family work without a code change. Non-floating
    dtypes are rejected because they cannot describe a parameter's precision.
    """
    if value is None:
        return None
    if isinstance(value, str):
        resolved = getattr(torch, value.removeprefix("torch."), None)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(
                f"Unknown dtype {value!r}. Expected the name of a torch floating dtype, e.g. 'bfloat16'."
            )
    elif isinstance(value, torch.dtype):
        resolved = value
    else:
        # ValueError, not TypeError: this doubles as draccus's decoder, and draccus only treats
        # ValueError as "this branch of the union does not match".
        raise ValueError(f"dtype must be a torch.dtype, the name of one, or None; got {value!r}.")
    if not resolved.is_floating_point:
        raise ValueError(f"dtype must be a floating-point dtype, got {resolved}.")
    return resolved


@draccus.encode.register
def _encode_dtype(value: torch.dtype) -> str:
    """Serialize ``torch.bfloat16`` as ``"bfloat16"`` so ``config.json`` stays plain JSON."""
    return str(value).removeprefix("torch.")


draccus.decode.register(torch.dtype, _coerce_dtype)


def _warn_torch_dtype_alias() -> None:
    warnings.warn(
        "`torch_dtype` is deprecated; use `dtype`. It still works, and it is still read from "
        "existing checkpoints, but it is no longer written.",
        FutureWarning,
        stacklevel=3,
    )


@dataclass
class DtypeConfigMixin:
    """Gives a config the single field that requests a model's parameter storage precision.

    Mixed into every config class whose model is a ``PreTrainedPolicy``-style ``nn.Module`` --
    today ``PreTrainedConfig`` and ``RewardModelConfig``.

    ``dtype`` is a *request*, never a record of what was built: nothing in LeRobot writes back to
    it. It is held as a real ``torch.dtype`` so callers never have to parse it, and serialized as
    its plain name (``"bfloat16"``). Assigning either form works, at construction time or later.

    ``None`` means "unspecified" -- the model is left exactly as its ``__init__`` built it. It does
    *not* mean float32.
    """

    dtype: torch.dtype | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        # Both directions of the deprecated alias are handled here rather than through a property,
        # so they do not depend on where `__setattr__` sits in a subclass's MRO.
        if name == "torch_dtype":
            _warn_torch_dtype_alias()
            name = "dtype"
        if name == "dtype":
            value = _coerce_dtype(value)
        super().__setattr__(name, value)

    @property
    def torch_dtype(self) -> torch.dtype | None:
        """Deprecated alias for `dtype`, for code that still uses the old spelling.

        Checkpoints are handled separately, and silently, by `_migrate_config_dict`: a deprecation
        warning must not fire on data the user may not control. This fires on code, which they do.

        No removal version is promised. transformers announced that its own `torch_dtype` would go
        in 4.59; it is still there, past v5, because the name lives in published artifacts.
        """
        _warn_torch_dtype_alias()
        return self.dtype


class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"
    LANGUAGE = "LANGUAGE"


class PipelineFeatureType(str, Enum):
    ACTION = "ACTION"
    OBSERVATION = "OBSERVATION"


class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"
    QUANTILES = "QUANTILES"
    QUANTILE10 = "QUANTILE10"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple[int, ...]


class RTCAttentionSchedule(str, Enum):
    ZEROS = "ZEROS"
    ONES = "ONES"
    LINEAR = "LINEAR"
    EXP = "EXP"
