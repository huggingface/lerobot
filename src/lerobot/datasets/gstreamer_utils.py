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

"""GStreamer encoding backend.

Every entry in :data:`lerobot.configs.video.HW_VIDEO_CODECS` is an FFmpeg
encoder name, reached through PyAV. Some platforms expose their video encoder
only as GStreamer elements and have no FFmpeg encoder at all, so no ``vcodec``
value can select them. This module is a second implementation behind
:attr:`~lerobot.configs.video.VideoEncoderConfig.video_backend` for those
platforms.

NVIDIA Jetson is the current case. Its encoder is provided by the
``nvvideo4linux2`` GStreamer plugin. PyAV reports ``h264_nvenc`` as available
there because it is compiled into the bundled FFmpeg, but opening it fails
with ``OpenEncodeSessionEx: unsupported device``, so ``vcodec="auto"`` on the
PyAV backend selects an encoder that cannot run rather than falling back to
software.

The output is an ordinary MP4, identical in kind to what the PyAV backend
produces. Only the way the encoder is driven differs.

Requires the GStreamer Python bindings (``python3-gi``,
``gir1.2-gstreamer-1.0``) and the platform's encoder plugins.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

GST_VIDEO_CODECS: dict[str, str] = {
    "nvv4l2h264enc": "h264parse",
    "nvv4l2h265enc": "h265parse",
    "nvv4l2av1enc": "av1parse",
}

GST_HW_CODEC_PREFERENCE: list[str] = ["nvv4l2av1enc", "nvv4l2h265enc", "nvv4l2h264enc"]

GST_BITRATE_ONLY_CODECS: frozenset[str] = frozenset({"nvv4l2av1enc"})

# The nvv4l2 encoders write the codec's parameter set ONCE, at the start of the
# stream: insert-sps-pps (H.264/H.265) and insert-seq-hdr (AV1) both default to
# false. The muxer still lists every intra frame in the container's stss table as
# a sync sample, so the file advertises random access it cannot honour -- a
# decoder that jumps to one of those frames has never seen the parameter set and
# fails with "Invalid data found when processing input".
#
# Playback is unaffected, because playing starts at frame 0 and picks the header
# up there. Only seeking breaks. That makes the failure invisible at record time
# and invisible in a dataset viewer, and it surfaces only when something trains
# on the data -- which is random access by definition.
GST_HEADER_INSERTION_PROPERTY: dict[str, str] = {
    "nvv4l2h264enc": "insert-sps-pps",
    "nvv4l2h265enc": "insert-sps-pps",
    "nvv4l2av1enc": "insert-seq-hdr",
}

# idrinterval defaults to 256 on these elements while iframeinterval defaults to
# 30. Repeating the parameter set only helps at IDR frames, so leaving the two
# mismatched means a seek still decodes up to 255 frames to reach its target.
# Matching them makes each intra frame a real random-access point.
GST_DEFAULT_GOP: int = 30

_EOS_TIMEOUT_NS = 30 * 1_000_000_000

_BPP_AT_CRF30 = 0.89

_MP4_TIMESCALE = 12000


def _gst_quote(value: Any) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


_GST_INITIALISED = False


def _gst():
    """Import and initialise GStreamer once, raising ImportError if unavailable."""
    global _GST_INITIALISED
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # noqa: N806
    except (ImportError, ValueError) as e:
        raise ImportError(
            "The GStreamer encoding backend requires the GStreamer Python bindings. "
            "On Debian/Ubuntu: apt install python3-gi gir1.2-gstreamer-1.0 "
            "gir1.2-gst-plugins-base-1.0. These are system packages, so a virtualenv "
            "must be created with --system-site-packages to see them."
        ) from e

    if not _GST_INITIALISED:
        Gst.init(None)
        _GST_INITIALISED = True
    return Gst


def is_gstreamer_available() -> bool:
    """Whether the GStreamer backend can be used on this machine."""
    try:
        _gst()
    except ImportError:
        return False
    return True


def detect_available_encoders_gst(encoders: list[str] | str) -> list[str]:
    """Return the subset of ``encoders`` that can be instantiated on this machine.

    Both registration and instantiation are checked, so an element whose plugin
    is installed but whose device is absent is reported as unavailable.
    """
    if isinstance(encoders, str):
        encoders = [encoders]
    try:
        gst = _gst()
    except ImportError:
        return []

    available = []
    for name in encoders:
        factory = gst.ElementFactory.find(name)
        if factory is None:
            continue
        element = factory.create(None)
        if element is None:
            continue
        available.append(name)
    return available


def gst_codec_options(
    vcodec: str,
    crf: int | float | None,
    preset: int | str | None,
    g: int | None,
    extra_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate encoder settings into GStreamer element properties.

    ``g`` maps to ``iframeinterval`` and ``idrinterval``, which are set together
    even when ``g`` is ``None`` so the element's mismatched defaults (30 and 256)
    do not make seeking needlessly slow.

    The codec's parameter-set repetition (``insert-sps-pps`` / ``insert-seq-hdr``)
    is enabled for every element in :data:`GST_HEADER_INSERTION_PROPERTY`. It is
    not optional: without it the file plays but cannot be seeked, so a dataset
    recorded through this backend cannot be trained on.

    ``crf`` maps to a fixed ``qp-range`` with ``control-rate=0`` on elements
    that expose a quantizer. Elements in :data:`GST_BITRATE_ONLY_CODECS` expose
    only ``bitrate``; for those the mapping needs the frame size and rate and is
    applied by :class:`GStreamerVideoWriter`. Pass
    ``extra_options={"bitrate": ...}`` to set a target explicitly.

    ``preset`` maps to ``preset-level``, accepting either the integer the
    element takes or one of ``ultrafast``/``fast``/``medium``/``slow``.
    """
    opts: dict[str, Any] = {}

    # Always set both, even when the caller passed no ``g``: the element defaults
    # (iframeinterval 30, idrinterval 256) disagree, and the gap costs seek time.
    gop = GST_DEFAULT_GOP if g is None else int(g)
    opts["iframeinterval"] = gop
    opts["idrinterval"] = gop

    # Repeat the parameter set at every IDR so the sync samples the muxer
    # advertises are genuinely decodable. Without this the output plays but
    # cannot be seeked, and therefore cannot be trained on.
    header_property = GST_HEADER_INSERTION_PROPERTY.get(vcodec)
    if header_property is not None:
        opts[header_property] = 1

    if vcodec not in GST_BITRATE_ONLY_CODECS and crf is not None:
        qp = max(0, min(51, int(crf)))
        opts["control-rate"] = 0
        opts["qp-range"] = f"{qp},{qp}:{qp},{qp}:{qp},{qp}"

    if preset is not None:
        named = {"ultrafast": 1, "fast": 2, "medium": 3, "slow": 4}
        opts["preset-level"] = int(preset) if not isinstance(preset, str) else named.get(preset, 4)

    for k, v in (extra_options or {}).items():
        opts.setdefault(k, v)
    return opts


def default_bitrate_for_crf(width: int, height: int, fps: float, crf: int | float | None) -> int:
    """Bitrate target for elements that expose no quantizer control.

    Scales bits-per-pixel with ``crf`` so the setting keeps its usual direction.
    The curve is anchored on the rate ``libsvtav1`` produces at the same ``crf``
    on camera footage, so switching a dataset to a bitrate-only encoder lands
    near the size it had before rather than silently much smaller.

    This is a heuristic, not an equivalence: the rate a quantizer produces
    depends on the content, and easy footage will undershoot the target. Set
    ``extra_options={"bitrate": ...}`` when the rate matters.
    """
    crf = 30 if crf is None else float(crf)
    bpp = _BPP_AT_CRF30 * (0.5 ** ((crf - 30.0) / 8.0))
    return max(100_000, int(width * height * fps * bpp))


class GStreamerVideoWriter:
    """Encode RGB frames to an MP4 through a GStreamer pipeline.

    Construct, :meth:`write` frames in order, then :meth:`close`. The pipeline is
    ``appsrc -> <converter> -> <encoder> -> <parser> -> mp4mux -> filesink``.

    Where ``nvvidconv`` is available it transfers frames into the memory the
    encoder reads from; otherwise the pipeline stays in system memory.
    """

    def __init__(
        self,
        video_path: str | Path,
        fps: int,
        width: int,
        height: int,
        vcodec: str,
        options: dict[str, Any] | None = None,
        crf: int | float | None = None,
    ):
        gst = _gst()
        self._gst = gst
        self.video_path = Path(video_path)
        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)
        self.vcodec = vcodec
        self.frame_count = 0
        self._closed = False
        self._use_nvmm = gst.ElementFactory.find("nvvidconv") is not None

        if vcodec not in GST_VIDEO_CODECS:
            raise ValueError(
                f"Unsupported GStreamer video codec {vcodec!r}. Supported: {sorted(GST_VIDEO_CODECS)}"
            )

        opts = dict(options or {})
        if vcodec in GST_BITRATE_ONLY_CODECS and "bitrate" not in opts:
            opts["bitrate"] = default_bitrate_for_crf(self.width, self.height, self.fps, crf)
            opts.setdefault("control-rate", 0)
        self.options = opts

        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        gst = self._gst
        parser = GST_VIDEO_CODECS[self.vcodec]
        props = " ".join(f"{k}={v}" for k, v in self.options.items())
        convert = "videoconvert ! video/x-raw,format=NV12 ! "
        if self._use_nvmm:
            convert += "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
        desc = (
            f"appsrc name=src is-live=false block=true format=time "
            f"caps=video/x-raw,format=RGB,width={self.width},height={self.height},"
            f"framerate={self.fps}/1 "
            f"! {convert}"
            f"{self.vcodec} {props} ! {parser} ! "
            f"mp4mux trak-timescale={_MP4_TIMESCALE} ! "
            f"filesink location={_gst_quote(self.video_path)} sync=false"
        )
        logger.debug("GStreamer encode pipeline: %s", desc)
        self.pipeline = gst.parse_launch(desc)
        self.appsrc = self.pipeline.get_by_name("src")
        self.appsrc.set_property("max-bytes", 8 * self.width * self.height * 3)

        ret = self.pipeline.set_state(gst.State.PLAYING)
        if ret == gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(gst.State.NULL)
            self._drain_errors(raise_on_error=True)
            raise RuntimeError(f"GStreamer pipeline failed to start for {self.video_path}")

    def _drain_errors(self, raise_on_error: bool = False) -> None:
        """Surface pipeline errors, which GStreamer reports on the bus."""
        gst = self._gst
        bus = self.pipeline.get_bus()
        while True:
            msg = bus.pop_filtered(gst.MessageType.ERROR | gst.MessageType.WARNING)
            if msg is None:
                return
            err, debug = msg.parse_error() if msg.type == gst.MessageType.ERROR else msg.parse_warning()
            text = f"GStreamer ({self.vcodec}): {err.message} [{debug}]"
            if msg.type == gst.MessageType.ERROR:
                if raise_on_error:
                    raise RuntimeError(text)
                logger.error(text)
            else:
                logger.warning(text)

    def write(self, frame_rgb: np.ndarray) -> None:
        """Push one HWC uint8 RGB frame."""
        if self._closed:
            raise RuntimeError("write() called after close()")
        gst = self._gst
        if frame_rgb.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"frame is {frame_rgb.shape[1]}x{frame_rgb.shape[0]}, "
                f"pipeline expects {self.width}x{self.height}"
            )
        data = np.ascontiguousarray(frame_rgb, dtype=np.uint8).tobytes()
        buf = gst.Buffer.new_wrapped(data)
        duration = int(gst.SECOND / self.fps)
        buf.pts = self.frame_count * duration
        buf.dts = buf.pts
        buf.duration = duration
        ret = self.appsrc.emit("push-buffer", buf)
        if ret != gst.FlowReturn.OK:
            self._drain_errors()
            raise RuntimeError(f"GStreamer push-buffer returned {ret!r} for {self.video_path}")
        self.frame_count += 1
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        """Encoders can fail asynchronously once data flows, which a successful
        ``push-buffer`` does not reveal. Polling the bus is non-blocking."""
        gst = self._gst
        msg = self.pipeline.get_bus().pop_filtered(gst.MessageType.ERROR)
        if msg is None:
            return
        err, debug = msg.parse_error()
        raise RuntimeError(
            f"GStreamer encoder {self.vcodec} failed for {self.width}x{self.height}: "
            f"{err.message} [{debug}]. Hardware encoders reject frame sizes below their "
            f"supported minimum; try a larger resolution or a different vcodec."
        )

    def close(self) -> None:
        """Flush the encoder and finalise the container. Idempotent."""
        if self._closed:
            return
        self._closed = True
        gst = self._gst
        self.appsrc.emit("end-of-stream")
        bus = self.pipeline.get_bus()
        msg = bus.timed_pop_filtered(_EOS_TIMEOUT_NS, gst.MessageType.EOS | gst.MessageType.ERROR)
        self.pipeline.set_state(gst.State.NULL)
        if msg is None:
            raise RuntimeError(f"GStreamer encode timed out finalising {self.video_path}")
        if msg.type == gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            raise RuntimeError(f"GStreamer encode failed: {err.message} [{debug}]")

    def __del__(self) -> None:
        """Return the pipeline to NULL if it is dropped without close().

        GStreamer requires elements to reach the NULL state before the final
        reference goes away. Dropping a PLAYING pipeline aborts the process, so
        a caller that raises between construction and close() would take the
        interpreter down with it rather than surface the error.
        """
        if getattr(self, "_closed", True):
            return
        pipeline = getattr(self, "pipeline", None)
        gst = getattr(self, "_gst", None)
        if pipeline is None or gst is None:
            return
        with contextlib.suppress(Exception):
            pipeline.set_state(gst.State.NULL)

    def __enter__(self) -> GStreamerVideoWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
