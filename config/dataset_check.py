#!/usr/bin/env python3
"""Read a recorded dataset back and check it is what was meant to be recorded.

    ./config/dataset-check.sh carlkesselman/xlerobot-test     newest recording of that name
    ./config/dataset-check.sh carlkesselman/xlerobot-test_20261015_213012

lerobot-record stamps every new dataset with the date and time
(`<repo_id>_YYYYMMDD_HHMMSS`), so a plain name picks the newest stamped one.

Checks, locally only (never the Hub):
  - episodes and frames, and seconds of data per episode
  - the features: joints, cameras, and base velocities if the base had any
  - every camera decodes, at the frame in the middle of the recording;
    those frames are written out as PNGs to look at - the head should
    have natural colour, like the wrists (red/blue swapped = BGR)
  - each camera's video file exists and is non-empty
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def resolve(root: Path, repo: str) -> Path | None:
    exact = root / repo
    if (exact / "meta" / "info.json").is_file():
        return exact
    parent, name = (root / repo).parent, Path(repo).name
    stamped = sorted(
        (p for p in parent.glob(f"{name}_*") if (p / "meta" / "info.json").is_file()),
        key=lambda p: p.name,
    )
    return stamped[-1] if stamped else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo_id")
    ap.add_argument("--root", default=str(Path.home() / ".cache/huggingface/lerobot"))
    ap.add_argument("--out", default="readback", help="directory for the decoded frames")
    args = ap.parse_args()

    root = Path(args.root)
    path = resolve(root, args.repo_id)
    if path is None:
        print(f"no dataset for {args.repo_id} under {root}.", file=sys.stderr)
        print("  present:", file=sys.stderr)
        user = (root / args.repo_id).parent
        for p in sorted(user.glob("*")) if user.is_dir() else []:
            print(f"    {p.relative_to(root)}", file=sys.stderr)
        return 1
    repo = str(path.relative_to(root))

    import cv2
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo, root=path)
    print(f"\ndataset : {repo}")
    print(f"path    : {path}")
    print(f"episodes: {ds.num_episodes}, {ds.num_frames} frames at {ds.fps} fps "
          f"= {ds.num_frames / ds.fps:.1f} s")

    skip = ("index", "frame_index", "episode_index", "task_index", "timestamp")
    feats = {k: v for k, v in ds.features.items() if k not in skip}
    cams = sorted(k for k in feats if k.startswith("observation.images."))
    print("\nfeatures:")
    for k in sorted(feats):
        v = feats[k]
        names = v.get("names")
        if isinstance(names, dict):
            names = next(iter(names.values()), None)
        detail = f"{len(names)} values" if isinstance(names, list) else str(v.get("shape"))
        print(f"  {k:<36} {v.get('dtype', ''):<8} {detail}")
    state_names = feats.get("observation.state", {}).get("names") or []
    vel = [n for n in state_names if isinstance(n, str) and n.endswith(".vel")]
    print(f"  base velocities: {', '.join(vel) if vel else 'none (base: none)'}")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    f = ds[ds.num_frames // 2]
    backend = getattr(ds, "video_backend", getattr(ds, "_video_backend", "?"))
    print(f"\nframe {ds.num_frames // 2} decoded (backend: {backend}):")
    ok = True
    for k in cams:
        img = (f[k].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        name = k.split(".")[-1]
        dst = out / f"{name}.png"
        cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        r, g, b = (img[..., i].mean() for i in range(3))
        print(f"  {name:<12} {img.shape[1]}x{img.shape[0]}  mean R {r:5.1f} G {g:5.1f} B {b:5.1f}  -> {dst.name}")
        if img.max() == 0:
            print("               ALL BLACK - camera or decode problem")
            ok = False
    print(f"  written to {out}")
    print("  look at the PNGs: same colour balance on all three. A head image with")
    print("  red and blue swapped is still recording BGR.")

    print("\nvideo files:")
    for k in cams:
        files = sorted((path / "videos" / k).rglob("*.mp4"))
        size = sum(p.stat().st_size for p in files)
        print(f"  {k:<36} {len(files)} file(s), {size / 1e6:6.1f} MB")
        if not files or size == 0:
            ok = False

    print("\n" + ("OK - recorded, readable, complete." if ok else "PROBLEMS above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
