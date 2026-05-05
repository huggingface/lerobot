#!/usr/bin/env python3
"""
XAI Phase 1c — P0-V Raw Attention Map for Video.

Processes MP4 video frame-by-frame to extract language→image attention maps,
then generates an overlay video showing where the model attends over time.

Dependencies:
  - xai_p0v_raw_attention.py: Core attention extraction functions
  - xai_utils.py: Model loading bootstrap
  - xai_feature_maps.py: Visualization helpers

Usage:
    python3 xai_p0v_raw_attention_video.py --video test.mp4 --dry-run
    python3 xai_p0v_raw_attention_video.py --video video.mp4 --instruction "pouring seeds"
"""

import argparse
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xai_utils  # noqa: E402

from xai_utils import (
    PROJECT_DIR, OUTPUT_DIR,
    load_vision_tower, load_image_pil, pil_to_tensor,
    report_vram, ensure_output_dir, DAVIT_INPUT_SIZE,
)
from xai_feature_maps import overlay_heatmap, apply_colormap


_INSTR_SEEDS = "pouring the sunflower seeds From the orange cup into the clear cup"
_INSTR_COFFEE = (
    "Pour coffee from the orange cup into the cup "
    "with the black-bordered letter D sticker."
)


def extract_attention_weights(encoder, inputs_embeds, attention_mask) -> tuple:
    """Extracts raw attention tensors from the Florence-2 encoder."""
    encoder = encoder.to(dtype=torch.float32)
    with torch.no_grad():
        encoder_output = encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
    return encoder_output.attentions


def tokenize_instruction(instruction: str, device: "torch.device") -> "tuple[torch.Tensor, int]":
    """Tokenizes instruction text with BART; falls back to BOS/EOS if unavailable."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        input_ids = tokenizer(instruction, return_tensors="pt")["input_ids"].to(device)
    except Exception as exc:
        print(f"  [WARN] Tokenizer unavailable ({exc}). Using dummy BOS+EOS tokens.")
        input_ids = torch.tensor([[0, 2]], dtype=torch.long, device=device)

    seq_len = input_ids.shape[1]
    print(f"  input_ids: {input_ids.shape}  L={seq_len}")
    return input_ids, seq_len


def encode_image(pil_img, model, device: "torch.device") -> "tuple[torch.Tensor, int]":
    """Encodes a PIL image through the DaViT vision tower + projection layers."""
    pixel_values = pil_to_tensor(pil_img, device)
    with torch.no_grad():
        image_features = model._encode_image(pixel_values)
    image_features = image_features.to(dtype=torch.float32)
    n_img = image_features.shape[1]
    if n_img != 50:
        print(f"  [WARN] Expected N_img=50, got {n_img}. "
              f"Spatial token slicing will use n_img-1={n_img-1} tokens.")
    return image_features, n_img


def compute_heatmaps(all_attentions: tuple, n_img: int, seq_len: int) -> list:
    """Computes language→image attention heatmaps for each encoder layer."""
    n_spatial = n_img - 1
    grid_side = int(n_spatial ** 0.5)
    assert grid_side * grid_side == n_spatial, (
        f"Spatial token count {n_spatial} is not a perfect square - "
        f"cannot reshape to grid. Check image_feature_source config."
    )

    heatmaps = []
    for attn_layer in all_attentions:
        attn = attn_layer[0]
        lang_to_img = attn[:, n_img:, :n_img]
        heatmap_1d = lang_to_img.mean(dim=(0, 1)).cpu().numpy()
        heatmap_2d = heatmap_1d[1:].reshape(grid_side, grid_side)
        heatmap_norm = ((heatmap_2d - heatmap_2d.min())
                        / (heatmap_2d.max() - heatmap_2d.min() + 1e-8))
        heatmaps.append(heatmap_norm)
    return heatmaps


def get_video_metadata(video_path: str) -> dict:
    """Read video metadata without opening for frame iteration."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    metadata = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return metadata


def read_video_frames(video_path: str, skip_frames: int = 1, max_frames: int | None = None):
    """
    Generator that yields frames from an MP4 video as PIL Images.
    Yields (frame_index, pil_image) tuples.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_idx = 0
    yielded_count = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % skip_frames == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            yield frame_idx, pil_image
            yielded_count += 1

            if max_frames is not None and yielded_count >= max_frames:
                break

        frame_idx += 1

    cap.release()


def process_single_frame(
    frame_pil: Image.Image,
    model,
    encoder,
    device: torch.device,
    cached_input_ids: torch.Tensor,
    cached_lang_embeds: torch.Tensor,
    seq_len: int,
    verbose: bool = False,
) -> tuple:
    """
    Processes a single PIL frame through the P0-V attention pipeline.
    Returns (agg_heatmap (7,7), normalized_entropy float).
    """
    import numpy as np

    if verbose:
        print(f"  Encoding frame...")

    image_features, n_img = encode_image(frame_pil, model, device)

    if verbose:
        print(f"  Building merged embeddings...")

    inputs_embeds, attention_mask = model._merge_input_ids_with_image_features(
        image_features, cached_lang_embeds
    )

    expected_len = n_img + seq_len
    assert inputs_embeds.shape == (1, expected_len, 1024), (
        f"Merged embeds shape mismatch: expected (1, {expected_len}, 1024), "
        f"got {inputs_embeds.shape}"
    )

    all_attentions = extract_attention_weights(encoder, inputs_embeds, attention_mask)
    heatmaps = compute_heatmaps(all_attentions, n_img, seq_len)
    agg_heatmap = np.mean(heatmaps, axis=0)

    p = agg_heatmap / (agg_heatmap.sum() + 1e-8)
    entropy = -np.sum(p * np.log(p + 1e-8))
    norm_entropy = entropy / np.log(agg_heatmap.size)

    if verbose:
        print(f"  Entropy: {norm_entropy:.3f}")

    del all_attentions, inputs_embeds, image_features
    torch.cuda.empty_cache()

    return agg_heatmap, norm_entropy


def render_annotated_frame(
    frame_pil: Image.Image,
    agg_heatmap: "np.ndarray",
    norm_entropy: float,
    frame_idx: int,
    total_frames: int,
    instruction: str,
    alpha: float = 0.55,
    cmap: str = "turbo",
    output_size: tuple = (640, 480),
) -> "np.ndarray":
    """
    Overlays a heatmap on the frame and draws text annotations.
    Returns a BGR image (H, W, 3) for cv2.VideoWriter.
    """
    import cv2
    import numpy as np

    h, w = output_size[1], output_size[0]
    model_h, model_w = DAVIT_INPUT_SIZE
    display_wh = (model_w, model_h)

    mean_norm = ((agg_heatmap - agg_heatmap.min())
                 / (agg_heatmap.max() - agg_heatmap.min() + 1e-8))
    heatmap_up = cv2.resize(mean_norm.astype(np.float32), display_wh, interpolation=cv2.INTER_CUBIC)

    overlay_224 = overlay_heatmap(frame_pil, heatmap_up, alpha, cmap)
    overlay = cv2.resize(overlay_224, (w, h), interpolation=cv2.INTER_CUBIC)

    overlay_bgr = overlay[..., ::-1].copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(0.8, 0.5 * (w / 640.0) ** 0.5))
    instr_scale = max(0.35, min(0.6, font_scale * 0.82))
    thickness = 1

    ent_color = (
        0,
        int(255 * (1 - norm_entropy)),
        int(255 * norm_entropy),
    )
    ent_text = f"Entropy: {norm_entropy:.3f}"
    ent_size = cv2.getTextSize(ent_text, font, font_scale, thickness)[0]
    ent_top = 8
    ent_bottom = int(ent_top + ent_size[1] + 10)
    cv2.rectangle(overlay_bgr, (w - ent_size[0] - 20, ent_top), (w - 5, ent_bottom), (0, 0, 0), -1)
    cv2.putText(
        overlay_bgr, ent_text,
        (w - ent_size[0] - 15, ent_bottom - 5),
        font, font_scale, ent_color, thickness,
    )

    max_instr_width = w - 20
    instr_text = instruction.strip()
    while instr_text:
        instr_size = cv2.getTextSize(instr_text, font, instr_scale, thickness)[0]
        if instr_size[0] <= max_instr_width:
            break
        if len(instr_text) <= 4:
            break
        instr_text = instr_text[:-2].rstrip()
    if instr_text != instruction.strip() and len(instr_text) > 3:
        instr_text = instr_text[:-3].rstrip() + "..."

    instr_size = cv2.getTextSize(instr_text, font, instr_scale, thickness)[0]
    instr_top = int(h - instr_size[1] - 14)
    instr_bottom = int(h - 5)
    cv2.rectangle(overlay_bgr, (5, instr_top), (instr_size[0] + 15, instr_bottom), (0, 0, 0), -1)
    cv2.putText(
        overlay_bgr, instr_text,
        (10, instr_bottom - 5),
        font, instr_scale, (255, 255, 255), thickness,
    )

    return overlay_bgr


def main(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np

    start_time = time.time()
    ensure_output_dir()

    print("=" * 60)
    print("P0-V Video Attention Map")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")

    print("\n[1] Loading model with encoder...")
    model = load_vision_tower(device, keep_language_encoder=True)
    encoder = model.language_model.model.encoder
    print(f"Model loaded. Encoder layers: {len(encoder.layers)}")
    report_vram(device, "after_load")

    print(f"\n[2] Reading video: {args.video}")
    metadata = get_video_metadata(args.video)
    total_frames = metadata["frame_count"]
    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]
    print(f"Video: {args.video} ({total_frames} frames, {fps:.1f} FPS, {width}x{height})")

    if args.dry_run:
        print("\n[Dry run] Exiting early.")
        return

    print("\n[3] Testing read_video_frames()...")
    test_frames = list(read_video_frames(args.video, skip_frames=1, max_frames=3))
    print(f"  Read {len(test_frames)} test frames")
    for idx, pil_img in test_frames:
        print(f"    Frame {idx}: {pil_img.size}")

    print("\n[4] Tokenizing instruction (cached for all frames)...")
    input_ids, seq_len = tokenize_instruction(args.instruction, device)
    lang_embeds = model.get_input_embeddings()(input_ids).to(dtype=torch.float32)
    print(f"  Instruction tokens: {seq_len}")

    print("\n[5] Testing process_single_frame() on first frame...")
    first_frame_idx, first_frame_pil = test_frames[0]
    agg_heatmap, norm_entropy = process_single_frame(
        first_frame_pil, model, encoder, device,
        input_ids, lang_embeds, seq_len, verbose=True
    )
    print(f"  Heatmap shape: {agg_heatmap.shape}")
    print(f"  Normalized entropy: {norm_entropy:.3f}")

    print("\n[6] Testing render_annotated_frame()...")
    annotated_bgr = render_annotated_frame(
        first_frame_pil, agg_heatmap, norm_entropy,
        first_frame_idx, total_frames, args.instruction,
        alpha=args.alpha, cmap=args.cmap
    )
    print(f"  Annotated frame shape: {annotated_bgr.shape}")

    print("\n[7] Processing video frames...")

    output_width, output_height = width, height
    output_fps = fps

    if args.episode is not None:
        output_filename = f"p0v_video_ep{args.episode:03d}.mp4"
    else:
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        output_filename = f"p0v_video_{video_name}.mp4"

    output_path = os.path.join(OUTPUT_DIR, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))

    if not writer.isOpened():
        print(f"  Error: Cannot create video writer: {output_path}")
        return

    print(f"  Output: {output_path}")
    print(f"  FPS: {output_fps}, Resolution: {output_width}x{output_height}")

    frame_generator = read_video_frames(args.video, skip_frames=1, max_frames=args.max_frames)

    processed_count = 0
    total_processed = args.max_frames if args.max_frames is not None else total_frames
    entropy_values = []
    start_loop_time = time.time()

    try:
        for frame_idx, frame_pil in frame_generator:
            agg_heatmap, norm_entropy = process_single_frame(
                frame_pil, model, encoder, device,
                input_ids, lang_embeds, seq_len, verbose=False
            )
            annotated_frame = render_annotated_frame(
                frame_pil, agg_heatmap, norm_entropy,
                frame_idx, total_frames, args.instruction,
                alpha=args.alpha, cmap=args.cmap,
                output_size=(output_width, output_height)
            )
            writer.write(annotated_frame)
            entropy_values.append(norm_entropy)
            processed_count += 1

            if processed_count % 10 == 0 or processed_count == total_processed:
                print(f"\r  Progress: {processed_count}/{total_processed} frames, "
                      f"Entropy: {norm_entropy:.3f}", end="")

    finally:
        writer.release()
        loop_time = time.time() - start_loop_time
        print()

    mean_entropy = np.mean(entropy_values) if entropy_values else 0.0
    print(f"\n  Processed {processed_count} frames in {loop_time:.1f}s")
    print(f"  Mean entropy: {mean_entropy:.3f}")
    print(f"  Output saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P0-V Raw Attention Map for Video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Training instructions for this model:\n"
            f"  seeds:  {_INSTR_SEEDS}\n"
            f"  coffee: {_INSTR_COFFEE}\n"
        ),
    )
    parser.add_argument(
        "--video",
        default="/home/tunx/XVLA/test_video_pick_cup/file-000-cam1.mp4",
        help="Input video path (.mp4)",
    )
    parser.add_argument(
        "--instruction",
        default=_INSTR_COFFEE,
        help="Instruction text (default: seeds instruction)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode number for output naming",
    )
    parser.add_argument(
        "--camera",
        default="camera1",
        help="Camera view (default: camera1)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Heatmap overlay opacity [0.0–1.0] (default: 0.55)",
    )
    parser.add_argument(
        "--cmap",
        default="turbo",
        help="Matplotlib colormap name (default: turbo)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Output video FPS (default: 15)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Deprecated: output now matches original video frame count (default: 1, ignored)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model and read video metadata only",
    )
    args = parser.parse_args()
    main(args)