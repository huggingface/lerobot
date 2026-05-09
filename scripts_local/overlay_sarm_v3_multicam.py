"""Overlay SARM progress on a recorded demo video — multi-cam ckpt support.

Loads 2-cam SARM ext model, runs inference using both front+wrist as input,
displays only `front` camera in the output video.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot_policy_sarm.configuration_sarm import SARMConfig
from lerobot_policy_sarm.processor_sarm import SARMEncodingProcessorStep
from lerobot_policy_sarm.sarm_utils import temporal_proportions_to_breakpoints


def draw_text(img, lines, x=16, y0=30, line_h=30, font_scale=0.7, thickness=2):
    for i, (text, color) in enumerate(lines):
        y = y0 + i * line_h
        cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)


def draw_progress_bar(img, progress, x, y, w, h, threshold=0.9, breakpoints=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 2)
    filled_w = int(max(0.0, min(1.0, progress)) * (w - 4))
    color = (60, 220, 60) if progress >= threshold else (50, 200, 220)
    cv2.rectangle(img, (x + 2, y + 2), (x + 2 + filled_w, y + h - 2), color, -1)
    tx = x + 2 + int(threshold * (w - 4))
    cv2.line(img, (tx, y - 4), (tx, y + h + 4), (255, 255, 255), 2)
    if breakpoints is not None:
        for bp in breakpoints[1:-1]:
            bx = x + 2 + int(float(bp) * (w - 4))
            cv2.line(img, (bx, y - 2), (bx, y + h + 2), (180, 180, 180), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--pretrained", required=True)
    p.add_argument("--task", default="Three-stage assembly")
    p.add_argument("--stats", default=None)
    p.add_argument("--display-image-key", default="observation.images.front")
    p.add_argument("--out", default="outputs/sarm_videos")
    p.add_argument("--upscale", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--success-threshold", type=float, default=0.9)
    p.add_argument("--head-mode", default="sparse", choices=["sparse", "dense"])
    p.add_argument("--episodes", default=None)
    p.add_argument("--episode-labels", default=None)
    args = p.parse_args()

    episode_labels = {}
    if args.episode_labels:
        for kv in args.episode_labels.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                episode_labels[int(k.strip())] = v.strip()

    ds = LeRobotDataset(repo_id=args.dataset)
    print(f"dataset: {args.dataset}  eps={ds.num_episodes}  frames={ds.num_frames}  fps={ds.fps}")

    # Load SARM ext model directly via plugin (multi-cam aware)
    import json
    with open(Path(args.pretrained) / "config.json") as f:
        cfg_dict = json.load(f)
    cfg_dict.pop("type", None)
    # Trim fields not in SARMConfig dataclass
    valid_fields = {f.name for f in SARMConfig.__dataclass_fields__.values()}
    cfg_kwargs = {k: v for k, v in cfg_dict.items() if k in valid_fields}
    cfg = SARMConfig(**cfg_kwargs)
    cfg.device = args.device
    image_keys = list(cfg.image_keys or [cfg.image_key])
    state_key = cfg.state_key
    print(f"SARM image_keys: {image_keys}, state_key: {state_key}")
    print(f"Display: {args.display_image_key}")

    from lerobot_policy_sarm.modeling_sarm import SARMRewardModel
    from lerobot_policy_sarm.processor_sarm import make_sarm_ext_pre_post_processors
    stats_ds = LeRobotDataset(repo_id=args.stats) if args.stats else ds
    model = SARMRewardModel.from_pretrained(args.pretrained).to(args.device).eval()
    model.config.device = args.device
    preprocess, _ = make_sarm_ext_pre_post_processors(
        config=model.config, dataset_stats=stats_ds.meta.stats, dataset_meta=stats_ds.meta,
    )
    for s in getattr(preprocess, "steps", []):
        if hasattr(s, "eval"):
            s.eval()
    n_obs = cfg.n_obs_steps
    center_idx = n_obs // 2
    delta = list(cfg.observation_delta_indices)

    names_attr = f"{args.head_mode}_subtask_names"
    props_attr = f"{args.head_mode}_temporal_proportions"
    stage_names = list(getattr(cfg, names_attr, None) or ["task"])
    print(f"SARM {args.head_mode} head: {len(stage_names)} stages = {stage_names}")
    props = getattr(cfg, props_attr, None) or [1.0]
    breakpoints = np.array(temporal_proportions_to_breakpoints(props, len(props)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = int(ds.fps)
    ep_list = [int(x) for x in args.episodes.split(",")] if args.episodes else list(range(ds.num_episodes))

    for ep in ep_list:
        m = ds.meta.episodes[ep]
        a, b = int(m["dataset_from_index"]), int(m["dataset_to_index"])
        ep_len = b - a

        # Preload episode for both cams + state
        cam_tensors = {k: torch.stack([ds[a + j][k] for j in range(ep_len)], dim=0) for k in image_keys}
        state_all = torch.stack([ds[a + j][state_key] for j in range(ep_len)], dim=0)
        disp_all = ds[a]["__placeholder__"] if False else None  # placeholder
        # We'll re-fetch display image per frame from cam_tensors (front)

        writer = None
        for k_idx, i in enumerate(range(a, b)):
            window_idx = [max(0, min(ep_len - 1, k_idx + d)) for d in delta]
            per_cam = {kk: cam_tensors[kk][window_idx] for kk in image_keys}
            state_stack = state_all[window_idx]
            batch = {**per_cam, state_key: state_stack, "task": args.task,
                     "index": 0, "episode_index": 0}
            with torch.inference_mode():
                processed = preprocess(batch)
                prog, stage_probs = model.calculate_rewards(
                    text_embeddings=processed["text_features"],
                    video_embeddings=processed["video_features"],
                    state_features=processed.get("state_features"),
                    lengths=processed.get("lengths"),
                    frame_index=center_idx,
                    return_all_frames=False,
                    return_stages=True,
                    head_mode=args.head_mode,
                )
            if isinstance(prog, torch.Tensor):
                prog = float(prog.detach().cpu().reshape(-1)[0].item())
            else:
                prog = float(np.asarray(prog).reshape(-1)[0])
            prog = max(0.0, min(1.0, prog))

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

            # Display: front camera image only
            disp_img = cam_tensors[args.display_image_key][k_idx].cpu().numpy()
            if disp_img.dtype != np.uint8:
                disp_img = (disp_img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            else:
                disp_img = disp_img.transpose(1, 2, 0)
            disp_img = cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR)
            scale = args.upscale
            H, W = disp_img.shape[:2]
            frame = cv2.resize(disp_img, (W * scale, H * scale), interpolation=cv2.INTER_CUBIC)

            success = prog >= args.success_threshold
            marker = "SUCCESS" if success else "-"
            marker_color = (60, 220, 60) if success else (200, 200, 200)
            ep_label = episode_labels.get(ep, "")
            header = f"ep {ep:2d} frame {k_idx:3d}/{ep_len - 1}"
            if ep_label:
                header += f" [{ep_label}]"
            lines = [
                (header, (255, 255, 255)),
                (f"stage: {stage_name} ({stage_conf:.2f})", (220, 220, 80)),
                (f"progress: {prog:.3f}", (80, 200, 240)),
                (f"[{marker}]", marker_color),
            ]
            draw_text(frame, lines, x=16, y0=30, line_h=28, font_scale=0.7, thickness=2)

            bar_h = 20
            bar_y = frame.shape[0] - bar_h - 12
            bar_x = 16
            bar_w = frame.shape[1] - 32
            draw_progress_bar(frame, prog, bar_x, bar_y, bar_w, bar_h,
                              threshold=args.success_threshold, breakpoints=breakpoints)

            if writer is None:
                label = episode_labels.get(ep, f"ep{ep:02d}")
                out_path = out_dir / f"{label}_ep{ep:02d}.mp4"
                writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                            quality=8, pixelformat="yuv420p", macro_block_size=1)
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if writer is not None:
            writer.close()
            print(f"  ep {ep}: wrote {out_path} (len={ep_len})")

    print(f"\nAll videos in {out_dir}")


if __name__ == "__main__":
    main()
