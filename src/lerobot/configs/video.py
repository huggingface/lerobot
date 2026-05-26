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
from typing import Any, ClassVar, Self

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
VALID_VIDEO_CODECS: frozenset[str] = frozenset(
    {"h264", "hevc", "libsvtav1", "ffv1", "auto", *HW_VIDEO_CODECS}
)
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

# Default depth quantization and encoding parameters.
DEPTH_QUANT_BITS: int = 12
DEPTH_QMAX: int = (1 << DEPTH_QUANT_BITS) - 1  # 4095

DEFAULT_DEPTH_MIN: float = 0.01
DEFAULT_DEPTH_MAX: float = 10.0
DEFAULT_DEPTH_SHIFT: float = 3.5
DEFAULT_DEPTH_USE_LOG: bool = True
DEFAULT_DEPTH_PIX_FMT: str = "gray12le"

# Depth-specific tuning fields persisted under ``features[*]["info"]`` as ``video.<name>``.
DEPTH_ENCODER_INFO_FIELD_NAMES: frozenset[str] = frozenset({"depth_min", "depth_max", "shift", "use_log"})


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

    # Source-data channel count this encoder is expected to handle (3 for RGB,
    # 1 for depth, etc.)
    _DEFAULT_CHANNELS: ClassVar[int] = 3

    def __post_init__(self) -> None:
        self.resolve_vcodec()
        # Empty-constructor ergonomics: ``VideoEncoderConfig()`` must "just work".
        if self.preset is None and self.vcodec == "libsvtav1":
            self.preset = LIBSVTAV1_DEFAULT_PRESET
        self.validate()

    @classmethod
    def _kwargs_from_video_info(cls, video_info: dict | None) -> dict[str, Any]:
        """Parse the ``video.*`` keys of a feature ``info`` block into
        constructor kwargs.
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

        return kwargs

    @classmethod
    def from_video_info(cls, video_info: dict | None) -> Self:
        """Reconstruct an encoder config from a video feature's ``info`` block.

        Missing or ``None`` values fall back to the class defaults. 
        """
        return cls(**cls._kwargs_from_video_info(video_info))

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

            check_video_encoder_parameters_pyav(
                self.vcodec, self.pix_fmt, self.get_codec_options(), channels=self._DEFAULT_CHANNELS
            )

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
        elif self.vcodec == "ffv1":
            # Lossless intra-frame codec. ``crf``/``preset``/``fast_decode``
            # are not meaningful.
            set_if("threads", encoder_threads)
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


@dataclass
class DepthEncoderConfig(VideoEncoderConfig):
    """Encoder configuration for depth-map streams.

    Inherits the full :class:`VideoEncoderConfig` surface (codec, GOP, CRF,
    preset, ``extra_options``…) and adds the four parameters of the depth
    quantizer.

    Defaults flip ``vcodec`` to ``"hevc"`` (Main 12 profile) and ``pix_fmt``
    to ``"gray12le"``.


    Attributes:
        depth_min: Minimum depth in physical units (e.g. metres) represented
            by quantum ``0``.
        depth_max: Maximum depth represented by quantum :data:`DEPTH_QMAX`.
        shift: Pre-log offset for numerical stability near zero.
        use_log: ``True`` for logarithmic quantization (default; matches
            sensor error profile), ``False`` for linear.
    """

    vcodec: str = "hevc"
    pix_fmt: str = "gray12le"

    depth_min: float = DEFAULT_DEPTH_MIN
    depth_max: float = DEFAULT_DEPTH_MAX
    shift: float = DEFAULT_DEPTH_SHIFT
    use_log: bool = DEFAULT_DEPTH_USE_LOG

    _DEFAULT_CHANNELS: ClassVar[int] = 1

    @classmethod
    def _kwargs_from_video_info(cls, video_info: dict | None) -> dict[str, Any]:
        """Layer the depth-specific tuning (``depth_min`` / ``depth_max`` /
        ``shift`` / ``use_log``) on top of the base parser. Missing keys
        fall back to the class defaults.
        """
        kwargs = super()._kwargs_from_video_info(video_info)
        video_info = video_info or {}
        for name in DEPTH_ENCODER_INFO_FIELD_NAMES:
            value = video_info.get(f"video.{name}")
            if value is not None:
                kwargs[name] = value
        return kwargs


def depth_encoder_defaults() -> DepthEncoderConfig:
    """Return a :class:`DepthEncoderConfig` with depth-camera defaults."""
    return DepthEncoderConfig()


def encoder_config_from_video_info(video_info: dict | None) -> VideoEncoderConfig:
    """Build the appropriate encoder config from a feature's ``info`` block.

    Dispatches to :class:`DepthEncoderConfig` when the dict marks the feature
    as a depth map and to :class:`VideoEncoderConfig`
    otherwise.

    Args:
        video_info: A feature's ``info`` dict as persisted in ``info.json``,
            or ``None`` (treated as an empty dict).

    Returns:
        A :class:`DepthEncoderConfig` for depth features, otherwise a
        :class:`VideoEncoderConfig`.
    """
    video_info = video_info or {}
    is_depth = bool(video_info.get("is_depth_map") or video_info.get("video.is_depth_map"))
    cls: type[VideoEncoderConfig] = DepthEncoderConfig if is_depth else VideoEncoderConfig
    return cls.from_video_info(video_info)
