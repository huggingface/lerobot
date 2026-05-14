#!/usr/bin/env python

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
"""PyAV-based compatibility checks for :class:`VideoEncoderConfig`.

Centralises all :mod:`av` introspection of the bundled FFmpeg build.
Checks degrade to a no-op when the target codec isn't available locally.
"""

import functools
import logging
from typing import Any

import av

logger = logging.getLogger(__name__)

FFMPEG_NUMERIC_OPTION_TYPES = ("INT", "INT64", "UINT64", "FLOAT", "DOUBLE")
FFMPEG_INTEGER_OPTION_TYPES = ("INT", "INT64", "UINT64")


@functools.cache
def get_codec(vcodec: str) -> av.codec.Codec | None:
    """PyAV write-mode ``Codec`` for *vcodec*, or ``None`` if unavailable."""
    try:
        return av.codec.Codec(vcodec, "w")
    except Exception:
        return None


@functools.cache
def _get_codec_options_by_name(vcodec: str) -> dict[str, av.option.Option]:
    """Private-option name → PyAV ``Option`` for *vcodec* (empty if unavailable)."""
    codec = get_codec(vcodec)
    if codec is None:
        return {}
    return {opt.name: opt for opt in codec.descriptor.options}


@functools.cache
def _get_codec_video_formats(vcodec: str) -> tuple[str, ...]:
    """Pixel formats accepted by *vcodec* in PyAV's preferred order (empty if unknown)."""
    codec = get_codec(vcodec)
    if codec is None:
        return ()
    return tuple(fmt.name for fmt in (codec.video_formats or []))


def detect_available_encoders_pyav(encoders: list[str] | str) -> list[str]:
    """Return the subset of *encoders* available as video encoders in the local FFmpeg build.

    Each name is probed directly via :func:`get_codec`; input order is preserved.
    """
    if isinstance(encoders, str):
        encoders = [encoders]

    available: list[str] = []
    for name in encoders:
        codec = get_codec(name)
        if codec is not None and codec.type == "video":
            available.append(name)
        else:
            logger.debug("encoder '%s' not available as video encoder", name)
    return available


def _check_option_value(vcodec: str, label: str, value: Any, opt: av.option.Option) -> None:
    """Range-check numeric *value* and choice-check string *value* against *opt*."""
    type_name = opt.type.name
    if type_name in FFMPEG_NUMERIC_OPTION_TYPES:
        if isinstance(value, bool):
            raise ValueError(
                f"{label}={value!r} is not numeric; codec {vcodec!r} expects a number for this option."
            )
        elif isinstance(value, str):
            try:
                num_val = float(value)
            except ValueError as e:
                raise ValueError(
                    f"{label}={value!r} is not numeric; codec {vcodec!r} expects a number for this option."
                ) from e
        elif isinstance(value, (float, int)):
            num_val = value
        else:
            raise ValueError(
                f"{label}={value!r} is not numeric; codec {vcodec!r} expects a number for this option."
            )

        # Check integer type compatibility
        if type_name in FFMPEG_INTEGER_OPTION_TYPES and not num_val.is_integer():
            raise ValueError(
                f"{label}={num_val!r} must be an integer for codec {vcodec!r} "
                f"(FFmpeg option {opt.name!r} is {type_name}); float values are not allowed."
            )

        # Check numeric range compatibility
        lo, hi = float(opt.min), float(opt.max)
        if lo < hi and not (lo <= num_val <= hi):
            raise ValueError(
                f"{label}={num_val} is out of range for codec {vcodec!r}; must be in [{lo}, {hi}]"
            )

    elif type_name == "STRING":
        if isinstance(value, bool):
            raise ValueError(f"{label}={value!r} is not a valid string value for codec {vcodec!r}.")
        if isinstance(value, str):
            str_val = value
        elif isinstance(value, (int, float)):
            str_val = str(value)
        else:
            raise ValueError(f"{label}={value!r} has unsupported type for STRING option on codec {vcodec!r}")

        # Check string choice compatibility
        choices = [c.name for c in (opt.choices or [])]
        if choices and str_val not in choices:
            raise ValueError(
                f"{label}={str_val!r} is not a supported choice for codec "
                f"{vcodec!r}; valid choices: {choices}"
            )
    else:
        return


def _check_pixel_format(vcodec: str, pix_fmt: str) -> None:
    formats = _get_codec_video_formats(vcodec)
    if formats and pix_fmt not in formats:
        raise ValueError(
            f"pix_fmt={pix_fmt!r} is not supported by codec {vcodec!r}; "
            f"supported pixel formats: {list(formats)}"
        )


def _check_codec_options(vcodec: str, codec_options: dict[str, Any]) -> None:
    """Validate merged encoder options (typed) against the codec's published AVOptions."""
    supported_options = _get_codec_options_by_name(vcodec)
    for key, value in codec_options.items():
        # GOP size is not a codec-specific option, it has to be validated separately.
        if key == "g":
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"g={value!r} must be a positive integer for codec {vcodec!r}")
            continue
        if key not in supported_options:
            continue
        _check_option_value(vcodec, key, value, supported_options[key])


def check_video_encoder_parameters_pyav(vcodec: str, pix_fmt: str, codec_options: dict[str, Any]) -> None:
    """Verify *config* is compatible with the bundled FFmpeg build.

    Checks pixel format, abstract tuning-field compatibility, and each merged
    encoder option from :meth:`~lerobot.configs.video.VideoEncoderConfig.get_codec_options`
    against PyAV (including numeric ``extra_options`` present in that dict).
    No-op when ``config.vcodec`` isn't in the local FFmpeg build.

    Raises:
        ValueError: on the first incompatibility encountered.
    """
    options = _get_codec_options_by_name(vcodec)
    if not options:
        raise ValueError(f"Codec {vcodec!r} is not available in the bundled FFmpeg build")
    _check_pixel_format(vcodec, pix_fmt)
    _check_codec_options(vcodec, codec_options)
