# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Isolate the streaming video-decode path — no SLURM, no DataLoader, no benchmark loop.

Reproduces exactly what StreamingLeRobotDataset does for one video (resolve path -> fsspec.open ->
torchcodec VideoDecoder -> get one frame) and prints the environment + the first bytes of the handle, so
a decode failure ("No valid stream found in input file") can be pinpointed: bad/placeholder bytes vs a
torchcodec/ffmpeg build issue vs a device issue.

    python benchmarks/streaming/diagnose_decode.py --repo_id pepijn223/robocasa_pretrain_human300_v4
    python benchmarks/streaming/diagnose_decode.py --repo_id … --data_files_root hf://buckets/<o>/<n>
    python benchmarks/streaming/diagnose_decode.py --repo_id … --video_decode_device cuda
"""

import argparse
import importlib.metadata as im

import fsspec

from lerobot.datasets import LeRobotDatasetMetadata


def _version(pkg: str) -> str:
    try:
        return im.version(pkg)
    except Exception:
        return "MISSING"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo_id", required=True)
    p.add_argument("--data_files_root", default=None, help="e.g. hf://buckets/<owner>/<name>")
    p.add_argument("--revision", default=None)
    p.add_argument("--video_decode_device", default="cpu")
    p.add_argument("--episode", type=int, default=0)
    args = p.parse_args()

    print("== environment ==")
    for pkg in ("torchcodec", "av", "huggingface_hub", "hf_xet", "datasets", "fsspec"):
        print(f"  {pkg}: {_version(pkg)}")

    meta = LeRobotDatasetMetadata(args.repo_id, revision=args.revision)
    video_key = meta.video_keys[0]
    rel_path = meta.get_video_file_path(args.episode, video_key)
    root = args.data_files_root.rstrip("/") if args.data_files_root else meta.url_root
    video_path = f"{root}/{rel_path}"
    print("\n== target ==")
    print(f"  video_key: {video_key}")
    print(f"  video_path: {video_path}")

    print("\n== fsspec handle ==")
    try:
        fh = fsspec.open(video_path).__enter__()
        head = fh.read(32)
        print(f"  first 32 bytes (hex): {head.hex()}")
        # A valid MP4/MOV has an 'ftyp' box near the start; anything else (HTML/JSON/empty) means the
        # handle resolved to a placeholder or error page, not the video bytes.
        looks_mp4 = b"ftyp" in head
        print(f"  looks like MP4 (contains 'ftyp'): {looks_mp4}")
        if not looks_mp4:
            print(f"  !! first bytes as text: {head[:32]!r}")
        fh.seek(0)
    except Exception as e:
        print(f"  !! fsspec.open/read FAILED: {type(e).__name__}: {e}")
        return

    print("\n== torchcodec VideoDecoder ==")
    try:
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(fh, seek_mode="approximate", device=args.video_decode_device)
        md = decoder.metadata
        print(f"  OK: {md.num_frames} frames, {md.average_fps} fps, codec={getattr(md, 'codec', '?')}")
        frame = decoder.get_frames_at(indices=[0])
        print(f"  decoded frame 0: shape={tuple(frame.data.shape)}, device={frame.data.device}")
        print("\nDECODE OK — the streaming pipeline can read this video on this machine.")
    except Exception as e:
        print(f"  !! VideoDecoder FAILED: {type(e).__name__}: {e}")
        print(
            "\nDECODE FAILED. If the bytes above look like MP4 (ftyp=True), this is a torchcodec/ffmpeg "
            "build issue, NOT bad bytes. Common cause for LeRobot v3 datasets: the videos are AV1-encoded "
            "(see the 'codec' line on a working machine). Then:\n"
            "  - CPU decode needs an ffmpeg built with an AV1 decoder (libdav1d/libaom); a build without it "
            "reports 'No valid stream found'.\n"
            "  - GPU/NVDEC decode of AV1 requires an Ada-generation GPU or newer (RTX 40 / L4 / L40). "
            "Ampere/Volta (A100/V100) NVDEC cannot decode AV1, so the decoder opens but yields 0 frames.\n"
            "Fix: install an AV1-capable ffmpeg/torchcodec on the node (and use an Ada+ GPU for --video_"
            "decode_device cuda), or re-encode the dataset to H.264/H.265.\n"
            "If ftyp=False instead, the handle resolved to a placeholder/error page (auth, revision, or Xet "
            "resolution) rather than the video bytes."
        )


if __name__ == "__main__":
    main()
