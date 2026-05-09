"""Overlay SARM progress + stage + success marker on a recorded demo video.

Ported from lerobot-panda/scripts/overlay_sarm_video.py; adapted to this
repo's SARM API (`lerobot.processor.reward_model.sarm`).

Walks a LeRobotDataset episode-by-episode, re-runs SARM on every frame
(using `return_stages=True`) to read per-frame stage probs + progress,
and emits one mp4 per episode with text drawn over the image frame.

Usage:
    uv run python -m lerobot.policies.sarm.overlay_sarm_video_sim_assemble \\
        --dataset domrachev03/sim_assemble_sarm_multistage_two_stages_filtered_v2_val \\
        --pretrained outputs/sim_assemble_sarm_multistage_iter4/checkpoints/last/pretrained_model \\
        --task "Two-stage assembly" \\
        --stats domrachev03/sim_assemble_sarm_multistage_two_stages_filtered_v2 \\
        --out outputs/sarm_videos/iter4 \\
        --episodes 0,6,7,8,9

The `--episodes` flag selects specific episode indices from the dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.reward_model.sarm import (
    SARMRewardConfig,
    SARMRewardProcessorStep,
)


def draw_text(
    img: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
    x: int = 16,
    y0: int = 30,
    line_h: int = 30,
    font_scale: float = 0.7,
    thickness: int = 2,
) -> None:
    for i, (text, color) in enumerate(lines):
        y = y0 + i * line_h
        cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)


def draw_progress_bar(
    img: np.ndarray,
    progress: float,
    x: int, y: int, w: int, h: int,
    threshold: float = 0.9,
    breakpoints: np.ndarray | None = None,
) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 2)
    filled_w = int(max(0.0, min(1.0, progress)) * (w - 4))
    color = (60, 220, 60) if progress >= threshold else (50, 200, 220)
    cv2.rectangle(img, (x + 2, y + 2), (x + 2 + filled_w, y + h - 2), color, -1)
    tx = x + 2 + int(threshold * (w - 4))
    cv2.line(img, (tx, y - 4), (tx, y + h + 4), (255, 255, 255), 2)
    # breakpoint ticks (grey)
    if breakpoints is not None:
        for bp in breakpoints[1:-1]:
            bx = x + 2 + int(float(bp) * (w - 4))
            cv2.line(img, (bx, y - 2), (bx, y + h + 2), (180, 180, 180), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--task", default="Two-stage assembly")
    parser.add_argument("--stats", default=None)
    parser.add_argument("--display-image-key", default="observation.images.front",
                        help="Image to display in the video (different from model's image_key).")
    parser.add_argument("--out", default="outputs/sarm_videos")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--success-threshold", type=float, default=0.9)
    parser.add_argument("--head-mode", default="sparse", choices=["sparse", "dense"])
    parser.add_argument("--episodes", default=None,
                        help="Comma-separated episode indices (default: all).")
    parser.add_argument("--episode-labels", default=None,
                        help="Comma-separated 'ep_idx=label' pairs (e.g. 0=full,6=0of4).")
    args = parser.parse_args()

    episode_labels: dict[int, str] = {}
    if args.episode_labels:
        for kv in args.episode_labels.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                episode_labels[int(k.strip())] = v.strip()

    ds = LeRobotDataset(repo_id=args.dataset)
    print(f"dataset: {args.dataset}  eps={ds.num_episodes}  frames={ds.num_frames}  fps={ds.fps}")

    cfg = SARMRewardConfig(
        type="sarm",
        pretrained_path=args.pretrained,
        stats_dataset_repo_id=args.stats,
        device=args.device,
        task=args.task,
        head_mode=args.head_mode,
        reward_mode="dense",
    )
    step = SARMRewardProcessorStep(config=cfg, terminate_on_success=False)
    model = step._model
    preprocess = step._preprocess
    center_idx = step._center_idx

    names_attr = f"{args.head_mode}_subtask_names"
    props_attr = f"{args.head_mode}_temporal_proportions"
    stage_names = list(getattr(model.config, names_attr, None) or ["task"])
    print(f"SARM {args.head_mode} head: {len(stage_names)} stages = {stage_names}")
    print(f"SARM model image_key: {step._image_key} (display: {args.display_image_key})")

    # Compute breakpoints for progress-bar ticks.
    from lerobot.policies.sarm.sarm_utils import temporal_proportions_to_breakpoints
    props = getattr(model.config, props_attr, None) or [1.0]
    breakpoints = np.array(temporal_proportions_to_breakpoints(props, len(props)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = int(ds.fps)

    if args.episodes:
        ep_list = [int(x) for x in args.episodes.split(",")]
    else:
        ep_list = list(range(ds.num_episodes))

    for ep in ep_list:
        m = ds.meta.episodes[ep]
        a, b = int(m["dataset_from_index"]), int(m["dataset_to_index"])
        step.reset()
        writer = None

        for k, i in enumerate(range(a, b)):
            f = ds[i]
            obs = {
                step._image_key: f[step._image_key],
                step._state_key: f[step._state_key],
            }
            step._push_obs_to_buffer(obs)
            img_snap, state_snap = step._snapshot_buffers()
            image_stack, state_stack = step._build_window_from_snapshot(img_snap, state_snap)
            batch = {step._image_key: image_stack, "task": args.task,
                     "index": 0, "episode_index": 0}
            if state_stack is not None:
                batch[step._state_key] = state_stack
            with torch.inference_mode():
                processed = preprocess(batch)
                progress, stage_probs = model.calculate_rewards(
                    text_embeddings=processed["text_features"],
                    video_embeddings=processed["video_features"],
                    state_features=processed.get("state_features"),
                    lengths=processed.get("lengths"),
                    frame_index=center_idx,
                    return_all_frames=False,
                    return_stages=True,
                    head_mode=args.head_mode,
                )
            if isinstance(progress, torch.Tensor):
                progress = float(progress.detach().cpu().reshape(-1)[0].item())
            else:
                progress = float(np.asarray(progress).reshape(-1)[0])
            progress = max(0.0, min(1.0, progress))

            if isinstance(stage_probs, torch.Tensor):
                stage_probs = stage_probs.detach().cpu().numpy()
            stage_probs = np.asarray(stage_probs)
            if stage_probs.ndim == 3:
                sp = stage_probs[0, center_idx]
            elif stage_probs.ndim == 2:
                sp = stage_probs[center_idx]
            else:
                sp = stage_probs
            stage_idx = int(np.argmax(sp))
            stage_name = stage_names[stage_idx] if stage_idx < len(stage_names) else f"stage_{stage_idx}"
            stage_conf = float(sp[stage_idx])

            # Display image (separate from SARM input image).
            disp_img = f[args.display_image_key].cpu().numpy()
            if disp_img.dtype != np.uint8:
                disp_img = (disp_img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            else:
                disp_img = disp_img.transpose(1, 2, 0)
            disp_img = cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR)

            scale = args.upscale
            H, W = disp_img.shape[:2]
            frame = cv2.resize(disp_img, (W * scale, H * scale), interpolation=cv2.INTER_CUBIC)

            success = progress >= args.success_threshold
            marker = "SUCCESS" if success else "-"
            marker_color = (60, 220, 60) if success else (200, 200, 200)
            ep_label = episode_labels.get(ep, "")
            header = f"ep {ep:2d} frame {k:3d}/{b - a - 1}"
            if ep_label:
                header = f"{header} [{ep_label}]"
            lines = [
                (header, (255, 255, 255)),
                (f"stage: {stage_name} ({stage_conf:.2f})", (220, 220, 80)),
                (f"progress: {progress:.3f}", (80, 200, 240)),
                (f"[{marker}]", marker_color),
            ]
            draw_text(frame, lines, x=16, y0=30, line_h=28, font_scale=0.7, thickness=2)

            bar_h = 20
            bar_y = frame.shape[0] - bar_h - 12
            bar_x = 16
            bar_w = frame.shape[1] - 32
            draw_progress_bar(frame, progress, bar_x, bar_y, bar_w, bar_h,
                              threshold=args.success_threshold,
                              breakpoints=breakpoints)

            if writer is None:
                label = episode_labels.get(ep, f"ep{ep:02d}")
                out_path = out_dir / f"{label}_ep{ep:02d}.mp4"
                writer = imageio.get_writer(
                    str(out_path),
                    fps=fps,
                    codec="libx264",
                    quality=8,
                    pixelformat="yuv420p",
                    macro_block_size=1,
                )
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if writer is not None:
            writer.close()
            print(f"  ep {ep}: wrote {out_path} (len={b - a})")

    print(f"\nAll videos written to {out_dir}")


if __name__ == "__main__":
    main()
