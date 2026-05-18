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
# Note: We subclass str so that serialization is straightforward
# https://stackoverflow.com/questions/24481852/serialising-an-enum-member-to-json

"""Video encoder configurations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from lerobot.utils.import_utils import require_package

logger = logging.getLogger(__name__)

# List of hardware encoders to probe for auto-selection. Availability depends on the platform and the chosen video backend.
# Determines the order of preference for auto-selection when vcodec="auto" is used.
HW_VIDEO_CODECS = [
    "h264_videotoolbox",  # macOS
    "hevc_videotoolbox",  # macOS
    "h264_nvenc",  # NVIDIA GPU
    "hevc_nvenc",  # NVIDIA GPU
    "h264_vaapi",  # Linux Intel/AMD
    "h264_qsv",  # Intel Quick Sync
]
VALID_VIDEO_CODECS: frozenset[str] = frozenset({"h264", "hevc", "libsvtav1", "auto", *HW_VIDEO_CODECS})
# Aliases for legacy video codec names.
VIDEO_CODECS_ALIASES: dict[str, str] = {"av1": "libsvtav1"}


LIBSVTAV1_DEFAULT_PRESET: int = 12

# Keys persisted under ``features[*]["info"]`` as ``video.<name>`` (from :class:`VideoEncoderConfig`).
# ``vcodec``` and ``pix_fmt`` are derived from the video stream directly.
VIDEO_ENCODER_INFO_FIELD_NAMES: frozenset[str] = frozenset(
    {"g", "crf", "preset", "fast_decode", "extra_options", "video_backend"}
)
VIDEO_ENCODER_INFO_KEYS: frozenset[str] = frozenset(
    f"video.{name}" for name in VIDEO_ENCODER_INFO_FIELD_NAMES
)


@dataclass
class VideoEncoderConfig:
    """Video encoder configuration.

    Attributes:
        vcodec: Video encoder name. ``"auto"`` is resolved during
            construction (HW encoder if available, else ``libsvtav1``).
        pix_fmt: Pixel format (e.g. ``"yuv420p"``).
        g: GOP size (keyframe interval).
        crf: Quality level — mapped to the native quality parameter of the
            codec (``crf`` for software, ``qp`` for NVENC/VAAPI,
            ``q:v`` for VideoToolbox, ``global_quality`` for QSV).
        preset: Speed/quality preset. Accepted type is per-codec.
        fast_decode: Fast-decode tuning. For ``libsvtav1`` this is a level (0-2)
            embedded in ``svtav1-params``. For ``h264`` and ``hevc`` non-zero values
            set ``tune=fastdecode``. Ignored for other codecs.
        video_backend: Python to be used for encoding. Only ``"pyav"``
            is currently supported.
        extra_options: Free-form dictionary of additional video encoder options
            (e.g. ``{"tune": "film", "profile:v": "high", "bf": 2}``).
    """

    vcodec: str = "libsvtav1"  # TODO(CarolinePascal): rename to codec ?
    pix_fmt: str = "yuv420p"
    g: int | None = 2
    crf: int | float | None = 30
    preset: int | str | None = None
    fast_decode: int = 0
    # TODO(CarolinePascal): add torchcodec support + find a way to unify the
    # two backends (encoding and decoding).
    video_backend: str = "pyav"
    extra_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.resolve_vcodec()
        # Empty-constructor ergonomics: ``VideoEncoderConfig()`` must "just work".
        if self.preset is None and self.vcodec == "libsvtav1":
            self.preset = LIBSVTAV1_DEFAULT_PRESET
        self.validate()

    @classmethod
    def from_video_info(cls, video_info: dict | None) -> VideoEncoderConfig:
        """Reconstruct a :class:`VideoEncoderConfig` from a video feature's ``info`` block.
        Missing or ``None`` values fall back to the class defaults.
        """
        video_info = video_info or {}
        kwargs: dict[str, Any] = {}

        for src_key, dst_field in (("video.codec", "vcodec"), ("video.pix_fmt", "pix_fmt")):
            value = video_info.get(src_key)
            if value is not None:
                kwargs[dst_field] = value

        for field_name in VIDEO_ENCODER_INFO_FIELD_NAMES:
            value = video_info.get(f"video.{field_name}")
            if value is None:
                continue
            # Persisted as ``{}`` after merges with disagreeing sources — treat as default.
            if field_name == "extra_options" and not value:
                continue
            kwargs[field_name] = value

        return cls(**kwargs)

    def detect_available_encoders(self, encoders: list[str] | str) -> list[str]:
        """Return the subset of available encoders based on the specified video backend.

        Args:
            encoders: List of encoder names to detect. If a string, it is converted to a list.
        Returns:
            List of available encoder names. If the video backend is not "pyav", returns an empty list.
        """
        if self.video_backend == "pyav":
            require_package("av", extra="dataset")
            from lerobot.datasets import detect_available_encoders_pyav

            return detect_available_encoders_pyav(encoders)
        return []

    def validate(self) -> None:
        """Validate the video encoder configuration."""
        if self.video_backend == "pyav":
            require_package("av", extra="dataset")
            from lerobot.datasets import check_video_encoder_parameters_pyav

            check_video_encoder_parameters_pyav(self.vcodec, self.pix_fmt, self.get_codec_options())

    def resolve_vcodec(self) -> None:
        """Check ``vcodec`` and, when it is ``"auto"``, pick a concrete encoder.

        For ``"auto"``, the first hardware encoder in the preference list that is available is chosen; if none are available, ``libsvtav1`` is used. If the
        resolved codec (explicit or after auto-selection) is not available, raises ``ValueError``.

        Stream-derived canonical codec names listed in :data:`VIDEO_CODECS_ALIASES` are
        rewritten to their corresponding encoder name (e.g. ``"av1"`` → ``"libsvtav1"``).
        """
        self.vcodec = VIDEO_CODECS_ALIASES.get(self.vcodec, self.vcodec)
        if self.vcodec not in VALID_VIDEO_CODECS:
            raise ValueError(f"Invalid vcodec '{self.vcodec}'. Must be one of: {sorted(VALID_VIDEO_CODECS)}")
        if self.vcodec == "auto":
            available = self.detect_available_encoders(HW_VIDEO_CODECS)
            for encoder in HW_VIDEO_CODECS:
                if encoder in available:
                    logger.info(f"Auto-selected video codec: {encoder}")
                    self.vcodec = encoder
                    return
            logger.warning("No hardware encoder available, falling back to software encoder 'libsvtav1'")
            self.vcodec = "libsvtav1"

        if self.detect_available_encoders(self.vcodec):
            logger.info(f"Using video codec: {self.vcodec}")
            return
        raise ValueError(f"Unsupported video codec: {self.vcodec} with video backend {self.video_backend}")

    def get_codec_options(
        self, encoder_threads: int | None = None, as_strings: bool = False
    ) -> dict[str, Any]:
        """Translate the tuning fields to codec-specific options.

        ``VideoEncoderConfig.extra_options`` are merged last but never override a structured field.

        Args:
            encoder_threads: Number of encoder threads set globally for all VideoEncoderConfigs.
                For libsvtav1, this is mapped to ``lp`` via ``svtav1-params``.
                For h264/hevc, this is mapped to ``threads``.
                Hardware encoders ignore this parameter.
            as_strings: If ``True``, casts values to strings.
        """
        opts: dict[str, Any] = {}

        def set_if(key: str, value: Any) -> None:
            if value is not None:
                opts[key] = value if not as_strings else str(value)

        # GOP size is not a codec-specific option, so it is always set.
        set_if("g", self.g)

        if self.vcodec == "libsvtav1":
            set_if("crf", self.crf)
            set_if("preset", self.preset)
            svtav1_parts: list[str] = []
            if self.fast_decode is not None:
                svtav1_parts.append(f"fast-decode={max(0, min(2, self.fast_decode))}")
            if encoder_threads is not None:
                svtav1_parts.append(f"lp={encoder_threads}")
            if svtav1_parts:
                opts["svtav1-params"] = ":".join(svtav1_parts)
        elif self.vcodec in ("h264", "hevc"):
            set_if("crf", self.crf)
            set_if("preset", self.preset)
            if self.fast_decode:
                opts["tune"] = "fastdecode"
            set_if("threads", encoder_threads)
        elif self.vcodec in ("h264_videotoolbox", "hevc_videotoolbox"):
            if self.crf is not None:
                opts["q:v"] = max(1, min(100, 100 - self.crf * 2))
        elif self.vcodec in ("h264_nvenc", "hevc_nvenc"):
            opts["rc"] = 0
            set_if("qp", self.crf)
            set_if("preset", self.preset)
        elif self.vcodec == "h264_vaapi":
            set_if("qp", self.crf)
        elif self.vcodec == "h264_qsv":
            set_if("global_quality", self.crf)
            set_if("preset", self.preset)
        else:
            set_if("crf", self.crf)
            set_if("preset", self.preset)

        # Extra options are merged last but never override structured fields (values are kept as given).
        for k, v in self.extra_options.items():
            if k not in opts:
                set_if(k, v)

        return opts


def camera_encoder_defaults() -> VideoEncoderConfig:
    """Return a :class:`VideoEncoderConfig` with RGB-camera defaults."""
    return VideoEncoderConfig()
