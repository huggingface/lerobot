"""Resize all videos in a HF lerobot dataset to a target HxW via ffmpeg.

Updates info.json features shape + video.height/width to match.
Usage:
    uv run python scripts_local/resize_videos.py \\
        --src-repo-id local/sim_3stage_v2_extra_combined \\
        --dst-repo-id local/sim_3stage_v2_extra_combined_128 \\
        --size 128
"""
import argparse, json, logging, shutil, subprocess
from pathlib import Path
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src-repo-id", required=True)
    ap.add_argument("--dst-repo-id", required=True)
    ap.add_argument("--size", type=int, required=True)
    args=ap.parse_args()
    src=Path(HF_LEROBOT_HOME)/args.src_repo_id
    dst=Path(HF_LEROBOT_HOME)/args.dst_repo_id
    if dst.exists(): shutil.rmtree(dst)
    logging.info(f"copy meta+data from {src} -> {dst}")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("videos"))
    cc=dst/"meta"/"clip_cache.npz"
    if cc.exists(): cc.unlink()

    # Walk source videos, re-encode each at target size
    for vid in sorted((src/"videos").rglob("*.mp4")):
        rel = vid.relative_to(src/"videos")
        out = dst/"videos"/rel
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg","-y","-i", str(vid),
               "-vf", f"scale={args.size}:{args.size}",
               "-c:v","libx264","-crf","18","-preset","fast",
               "-loglevel","error", str(out)]
        subprocess.run(cmd, check=True)
        logging.info(f"resized {rel}")

    # Update info.json image dims
    info=json.loads((dst/"meta"/"info.json").read_text())
    for k,v in info.get("features",{}).items():
        if k.startswith("observation.images."):
            v["shape"]=[3, args.size, args.size]
            v["video.height"]=args.size
            v["video.width"]=args.size
    (dst/"meta"/"info.json").write_text(json.dumps(info, indent=4))
    logging.info(f"DONE -> {dst}")


if __name__=="__main__": main()
