"""Verify that the GStreamer backend's DEFAULT settings produce seekable video.

This exercises the path a recording actually takes -- gst_codec_options() with no
explicit overrides -- rather than hand-passing properties, because the bug was
precisely that the defaults were wrong.

A dataset is only usable for training if frames can be fetched in shuffled order,
so "can I jump to an arbitrary frame" is the property under test, not "does it
play".
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/tk/lerobot/src")
from lerobot.datasets.gstreamer_utils import (  # noqa: E402
    GST_HEADER_INSERTION_PROPERTY,
    GStreamerVideoWriter,
    gst_codec_options,
)

W, H, FPS, N = 640, 360, 30, 240
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print("  {}  {}{}".format("PASS" if ok else "FAIL", name, ("  " + detail) if detail else ""))
    if not ok:
        FAILURES.append(name)


def make_frames(d: Path) -> None:
    rng = np.random.default_rng(0)
    for i in range(N):
        x = np.linspace(0, 255, W, dtype=np.float32)[None, :]
        y = np.linspace(0, 255, H, dtype=np.float32)[:, None]
        img = ((x + y + i * 3) % 256).astype(np.uint8)
        rgb = np.stack([img, np.roll(img, i, axis=1), np.roll(img, -i, axis=0)], -1)
        rgb = np.clip(rgb.astype(np.int16) + rng.integers(-8, 8, rgb.shape), 0, 255)
        Image.fromarray(rgb.astype(np.uint8)).save(d / f"frame-{i:06d}.png")


def seek_test(path: Path) -> tuple[bool, list[int]]:
    from torchcodec.decoders import VideoDecoder

    d = VideoDecoder(str(path))
    n = d.metadata.num_frames or N
    targets = [t for t in (31, 60, 97, 150, 200, n - 2) if 0 <= t < n]
    failed = []
    for t in targets:
        try:
            d.get_frames_at(indices=[t])
        except Exception:
            failed.append(t)
    # batched, the way the dataloader asks
    try:
        d.get_frames_at(indices=[t for t in (45, 46, 120, 199) if t < n])
    except Exception:
        failed.append(-1)
    return (not failed, failed)


def main() -> None:
    print("=== the option mapping is populated for every nvv4l2 encoder ===")
    for codec, prop in sorted(GST_HEADER_INSERTION_PROPERTY.items()):
        print(f"    {codec:<18} -> {prop}")
    check("all three nvv4l2 encoders covered", len(GST_HEADER_INSERTION_PROPERTY) == 3)

    print("=== defaults now request header repetition + a matched GOP ===")
    for codec in ("nvv4l2av1enc", "nvv4l2h264enc", "nvv4l2h265enc"):
        opts = gst_codec_options(codec, crf=30, preset=None, g=None)
        prop = GST_HEADER_INSERTION_PROPERTY[codec]
        check(f"{codec} sets {prop}", opts.get(prop) == 1, str(opts.get(prop)))
        check(
            f"{codec} idrinterval == iframeinterval",
            opts.get("idrinterval") == opts.get("iframeinterval"),
            "idr={} iframe={}".format(opts.get("idrinterval"), opts.get("iframeinterval")),
        )

    # extra_options is applied with setdefault, so it FILLS GAPS rather than
    # overriding anything the function computed -- the same treatment crf, preset
    # and g already get. Header insertion is therefore not disableable this way,
    # which is intended: without it the output cannot be trained on, and the
    # bitrate cost of repeating a header every 30 frames is a few dozen bytes.
    print("=== header insertion is not silently overridable (by design) ===")
    o = gst_codec_options("nvv4l2av1enc", crf=30, preset=None, g=None, extra_options={"insert-seq-hdr": 0})
    check(
        "extra_options cannot disable header insertion",
        o.get("insert-seq-hdr") == 1,
        str(o.get("insert-seq-hdr")),
    )
    o = gst_codec_options("nvv4l2av1enc", crf=30, preset=None, g=90)
    check("explicit g still honoured", o.get("idrinterval") == 90, str(o.get("idrinterval")))

    print("=== end-to-end: encode with the DEFAULTS and seek ===")
    tmp = Path(tempfile.mkdtemp(prefix="seekfix-"))
    imgs = tmp / "imgs"
    imgs.mkdir()
    make_frames(imgs)

    out = tmp / "default.mp4"
    opts = gst_codec_options("nvv4l2av1enc", crf=30, preset=None, g=None)
    with GStreamerVideoWriter(
        video_path=out, fps=FPS, width=W, height=H, vcodec="nvv4l2av1enc", options=opts, crf=30
    ) as wr:
        for p in sorted(imgs.glob("frame-*.png")):
            with Image.open(p) as im:
                wr.write(np.asarray(im.convert("RGB")))

    ok, failed = seek_test(out)
    check("seeking works with default options", ok, "" if ok else f"failed at {failed}")

    print("=== and the frames are real (not a decoder returning garbage) ===")
    from torchcodec.decoders import VideoDecoder

    d = VideoDecoder(str(out))
    a = d.get_frames_at(indices=[100]).data[0].float()
    b = d.get_frames_at(indices=[101]).data[0].float()
    seq = None
    for i, fr in enumerate(d):
        if i == 100:
            seq = fr.float()
            break
    diff_neighbour = float((a - b).abs().mean())
    diff_vs_seq = float((a - seq).abs().mean()) if seq is not None else 999.0
    check(
        "seeked frame matches the same frame read sequentially",
        diff_vs_seq < 2.0,
        f"mean abs diff {diff_vs_seq:.3f}",
    )
    check("consecutive frames actually differ", diff_neighbour > 0.5, f"mean abs diff {diff_neighbour:.3f}")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)}")
        for f in FAILURES:
            print("  -", f)
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
