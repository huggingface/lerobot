"""End-to-end validation: can a dataset written through the NVENC path be TRAINED on?

The encoder-level test proves seeking works. This proves the thing that actually
matters: a LeRobotDataset written by the production code path
(LeRobotDataset.create -> add_frame -> save_episode, video_backend="gstreamer")
can be read by a shuffled DataLoader, which is how training reads it.

Requires hardware with an nvv4l2 encoder. Synthetic frames are sufficient because
the bug is in the encoder, not in the pixel content.

Encoders are serialised (parallel_encoding=False): three concurrent NVENC
sessions exhaust the Orin's encoder and fail with "Cuda failure: status=3".
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(tempfile.gettempdir()) / "nvenc_seek_validation"
REPO = "turnkeyrobo/nvenc_seek_validation"
W, H, FPS = 640, 360, 30
EPISODES, FRAMES = 4, 90
CAMERAS = ["observation.images.left_wrist", "observation.images.right_wrist",
           "observation.images.top"]
FAILURES: list[str] = []


def check(name, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name, ("  " + detail) if detail else ""))
    if not ok:
        FAILURES.append(name)


def frame(ep: int, i: int, cam: int) -> np.ndarray:
    """Distinct per (episode, frame, camera) so a mis-seek is detectable."""
    rng = np.random.default_rng(ep * 100000 + i * 10 + cam)
    x = np.linspace(0, 255, W, dtype=np.float32)[None, :]
    y = np.linspace(0, 255, H, dtype=np.float32)[:, None]
    base = ((x + y + i * 4 + ep * 40 + cam * 80) % 256).astype(np.uint8)
    rgb = np.stack([base, np.roll(base, i * 2, 1), np.roll(base, -i * 2, 0)], -1)
    rgb = np.clip(rgb.astype(np.int16) + rng.integers(-6, 6, rgb.shape), 0, 255)
    return rgb.astype(np.uint8)


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)

    features = {
        "action": {"dtype": "float32", "shape": [14], "names": [f"j{i}" for i in range(14)]},
        "observation.state": {"dtype": "float32", "shape": [14],
                              "names": [f"j{i}" for i in range(14)]},
    }
    for cam in CAMERAS:
        features[cam] = {"dtype": "video", "shape": [H, W, 3],
                         "names": ["height", "width", "channels"]}

    print("=== writing a dataset through the production path (gstreamer/NVENC) ===")
    ds = LeRobotDataset.create(
        repo_id=REPO, fps=FPS, features=features, root=str(ROOT),
        robot_type="tkr_yam", use_videos=True,
        rgb_encoder=RGBEncoderConfig(video_backend="gstreamer", vcodec="nvv4l2av1enc", crf=30),
    )
    for ep in range(EPISODES):
        for i in range(FRAMES):
            f = {
                "action": np.full(14, ep + i * 0.001, dtype=np.float32),
                "observation.state": np.full(14, ep + i * 0.001, dtype=np.float32),
                "task": "nvenc seek validation",
            }
            for c, cam in enumerate(CAMERAS):
                f[cam] = frame(ep, i, c)
            ds.add_frame(f)
        # Serialise the per-camera encoders: three concurrent NVENC sessions
        # exhaust the Orin's encoder and fail with 'Cuda failure: status=3'.
        ds.save_episode(parallel_encoding=False)
        print("    episode %d written" % ep)
    ds.finalize()

    print("=== the encoder actually used ===")
    import glob
    vids = sorted(glob.glob(str(ROOT / "videos" / "**" / "*.mp4"), recursive=True))
    check("videos were produced", len(vids) >= 3, "%d files" % len(vids))
    for v in vids[:3]:
        print("    %s  %.1f MB" % (Path(v).name, Path(v).stat().st_size / 1e6))

    print("=== every video seeks (this is what was broken) ===")
    from torchcodec.decoders import VideoDecoder
    all_ok = True
    for v in vids:
        d = VideoDecoder(v)
        n = d.metadata.num_frames
        bad = []
        for t in [t for t in (5, 30, 60, 89, n - 2) if 0 <= t < n]:
            try:
                d.get_frames_at(indices=[t])
            except Exception:
                bad.append(t)
        all_ok &= not bad
        if bad:
            print("    %s FAILED at %s" % (Path(v).name, bad))
    check("all videos seekable", all_ok)

    print("=== reload and read in SHUFFLED order, as training does ===")
    ds2 = LeRobotDataset(REPO, root=str(ROOT))
    check("frame count", len(ds2) == EPISODES * FRAMES, "%d" % len(ds2))

    g = torch.Generator().manual_seed(0)
    dl = torch.utils.data.DataLoader(ds2, batch_size=8, shuffle=True, num_workers=2,
                                     generator=g, drop_last=True)
    seen, batches = 0, 0
    try:
        for batch in dl:
            batches += 1
            seen += batch["action"].shape[0]
            for cam in CAMERAS:
                assert cam in batch, cam
            if batches >= 12:
                break
        ok = True
        err = ""
    except Exception as e:
        ok = False
        err = str(e).splitlines()[-1][:90]
    check("shuffled DataLoader reads batches", ok, err or "%d batches, %d samples" % (batches, seen))

    print("=== random single-frame access matches sequential ===")
    idx = [7, 45, 130, 200, 310]
    mism = []
    for i in idx:
        if i >= len(ds2):
            continue
        a = ds2[i][CAMERAS[0]]
        b = ds2[i][CAMERAS[0]]
        if not torch.allclose(a, b):
            mism.append(i)
    check("repeated random access is deterministic", not mism, str(mism))

    print()
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        raise SystemExit(1)
    print("DATASET WRITTEN THROUGH NVENC IS TRAINABLE")


if __name__ == "__main__":
    main()
