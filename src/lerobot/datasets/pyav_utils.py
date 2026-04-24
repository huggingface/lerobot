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

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

import av

if TYPE_CHECKING:
    from lerobot.datasets.video_utils import VideoEncoderConfig

logger = logging.getLogger(__name__)

FFMPEG_NUMERIC_OPTION_TYPES = ("INT", "INT64", "UINT64", "FLOAT", "DOUBLE")

# Codec-specific FFmpeg private option whose value is controlled by the
# abstract ``crf`` tuning field.
CRF_OPTION_BY_CODEC: dict[str, str] = {
    "libsvtav1": "crf",
    "h264": "crf",
    "hevc": "crf",
    "h264_nvenc": "qp",
    "hevc_nvenc": "qp",
    "h264_vaapi": "qp",
    "h264_qsv": "global_quality",
}


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


def _is_field_supported(
    field_name: str, vcodec: str, options: dict[str, av.option.Option]
) -> bool:
    """Whether tuning option *field_name* is meaningful for *vcodec*."""
    # GOP is a stream-level option (AVStream.gop_size) not stored in private options.
    # Every video codec accepts it.
    if field_name == "g":
        return True
    if field_name == "crf":
        # Semantic "crf" maps to the codec's private option (see
        # CRF_OPTION_BY_CODEC), or to stream-level q:v for VideoToolbox.
        opt_name = CRF_OPTION_BY_CODEC.get(vcodec)
        return (opt_name is not None and opt_name in options) or vcodec in {
            "h264_videotoolbox",
            "hevc_videotoolbox",
        }
    if field_name == "fast_decode":
        # libsvtav1: svtav1-params:fast-decode=N — h264/hevc: tune=fastdecode.
        return "svtav1-params" in options or "tune" in options
    # preset and any future private-option-backed field: direct membership test.
    return field_name in options


def _check_numeric_range(
    label: str, num: float, opt: av.option.Option, vcodec: str
) -> None:
    """Raise if *num* lies outside *opt*'s numeric range (no-op if range is degenerate)."""
    lo, hi = float(opt.min), float(opt.max)
    if lo < hi and not (lo <= num <= hi):
        raise ValueError(
            f"{label}={num} is out of range for codec {vcodec!r}; must be in [{lo}, {hi}]"
        )


def _validate_option_value(
    vcodec: str, field_name: str, value: Any, opt: av.option.Option
) -> None:
    """Range-check numeric *value* and choice-check string *value* against *opt*.

    Type mismatches fall through to FFmpeg's own validation at encode time.
    """
    type_name = opt.type.name
    if type_name in FFMPEG_NUMERIC_OPTION_TYPES:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        _check_numeric_range(field_name, float(value), opt, vcodec)
    elif type_name == "STRING":
        if not isinstance(value, str):
            return
        choices = [c.name for c in (opt.choices or [])]
        if choices and value not in choices:
            raise ValueError(
                f"{field_name}={value!r} is not a supported choice for codec "
                f"{vcodec!r}; valid choices: {choices}"
            )
    else:
        return


def _validate_extra_option(
    vcodec: str, key: str, value: Any, opt: av.option.Option
) -> None:
    """Validate an ``extra_options`` entry: enforce numeric range/type only.

    Non-numeric options are passed through (FFmpeg accepts many ad-hoc strings).
    """
    if opt.type.name not in FFMPEG_NUMERIC_OPTION_TYPES:
        return

    label = f"extra_options[{key!r}]"
    not_numeric = ValueError(
        f"{label}={value!r} is not numeric; codec {vcodec!r} expects a number for this option."
    )
    if isinstance(value, bool):
        raise not_numeric
    if isinstance(value, (int, float)):
        num = float(value)
    elif isinstance(value, str):
        try:
            num = float(value)
        except ValueError as e:
            raise not_numeric from e
    else:
        raise not_numeric

    _check_numeric_range(label, num, opt, vcodec)


def _check_pixel_format(vcodec: str, pix_fmt: str, formats: tuple[str, ...]) -> None:
    if formats and pix_fmt not in formats:
        raise ValueError(
            f"pix_fmt={pix_fmt!r} is not supported by codec {vcodec!r}; "
            f"supported pixel formats: {list(formats)}"
        )


def _check_tuning_fields(
    config: VideoEncoderConfig, vcodec: str, options: dict[str, av.option.Option]
) -> None:
    tuning_options: tuple[str, ...] = config._TUNING_OPTIONS
    supported_fields = [f for f in tuning_options if _is_field_supported(f, vcodec, options)]
    for field_name in tuning_options:
        value = getattr(config, field_name)
        if not value:
            continue
        if field_name not in supported_fields:
            raise ValueError(
                f"{field_name}={value!r} is not supported by codec {vcodec!r}; "
                f"supported fields for this codec: {supported_fields}"
            )
        # Value shape is only cross-checkable when the field maps directly
        # to a private option: ``preset`` is literally ``"preset"``;
        # ``crf`` maps per-codec. ``g`` (stream-level) and ``fast_decode``
        # (composite) fall through to FFmpeg at encode time.
        if field_name == "preset":
            opt = options.get("preset")
        elif field_name == "crf":
            opt = options.get(CRF_OPTION_BY_CODEC.get(vcodec, ""))
        else:
            continue
        if opt is not None:
            _validate_option_value(vcodec, field_name, value, opt)


def _check_extra_options(
    config: VideoEncoderConfig, vcodec: str, options: dict[str, av.option.Option]
) -> None:
    # Torchcodec-style: only validate keys the codec exposes as AVOptions,
    # and only enforce numeric range / numeric-type. Everything else is
    # passed through (muxer options, ``x264-params``-style strings, etc.).
    for key, value in config.extra_options.items():
        opt = options.get(key)
        if opt is None:
            continue
        _validate_extra_option(vcodec, key, value, opt)


def check_video_encoder_config_pyav(config: VideoEncoderConfig) -> None:
    """Verify *config* is compatible with the bundled FFmpeg build.

    Checks pixel format, tuning-field availability, value range/choices for
    fields that map to a private option, and numeric ``extra_options``.
    No-op when ``config.vcodec`` isn't in the local FFmpeg build.

    Raises:
        ValueError: on the first incompatibility encountered.
    """
    vcodec = config.vcodec
    options = _get_codec_options_by_name(vcodec)
    if not options:
        logger.warning(
            "Codec %r is not available in the bundled FFmpeg build; ",
            vcodec,
        )
        return
    _check_pixel_format(vcodec, config.pix_fmt, _get_codec_video_formats(vcodec))
    _check_tuning_fields(config, vcodec, options)
    _check_extra_options(config, vcodec, options)
