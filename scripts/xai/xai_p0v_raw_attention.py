#!/usr/bin/env python3
"""
XAI Phase 1b — P0-V Raw Attention Map.

Extracts language→image attention weights from the Florence-2 encoder.
This reveals where the model attends when processing text instructions.

Dependencies:
  - xai_utils.py: Model loading bootstrap
  - xai_feature_maps.py: Visualization helpers

Usage:
    python3 xai_p0v_raw_attention.py --image test_image_2.jpg --instruction "Pour coffee"
    python3 xai_p0v_raw_attention.py --dry-run
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
    report_vram, ensure_output_dir,
)
from xai_feature_maps import overlay_heatmap, deletterbox_map, apply_colormap


def encode_image(pil_img, model, device: "torch.device") -> "tuple[torch.Tensor, int]":
    """Encodes a PIL image through the DaViT vision tower + projection layers."""
    pixel_values = pil_to_tensor(pil_img, device)
    print(f"  pixel_values: {pixel_values.shape}  dtype={pixel_values.dtype}")

    with torch.no_grad():
        image_features = model._encode_image(pixel_values)

    image_features = image_features.to(dtype=torch.float32)
    n_img = image_features.shape[1]
    print(f"  image_features: {image_features.shape}  dtype={image_features.dtype}  N_img={n_img}")

    if n_img != 50:
        print(f"  [WARN] Expected N_img=50, got {n_img}. "
              f"Spatial token slicing will use n_img-1={n_img-1} tokens.")

    return image_features, n_img


def tokenize_instruction(instruction: str, device: "torch.device") -> "tuple[torch.Tensor, int]":
    """
    Tokenizes an instruction string using the BART tokenizer.
    Falls back to dummy BOS+EOS tokens if the tokenizer is unavailable.
    """
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


def build_merged_embeds(
    model,
    image_features: "torch.Tensor",
    input_ids: "torch.Tensor",
    n_img: int,
    seq_len: int,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """
    Embeds language tokens and concatenates with image features.

    Token layout: [img_0 .. img_{N-1} | lang_0 .. lang_{L-1}]
    Returns inputs_embeds (1, n_img+L, 1024) and attention_mask (1, n_img+L).
    """
    lang_embeds = model.get_input_embeddings()(input_ids).to(dtype=torch.float32)
    print(f"  lang_embeds: {lang_embeds.shape}  dtype={lang_embeds.dtype}")

    inputs_embeds, attention_mask = model._merge_input_ids_with_image_features(
        image_features, lang_embeds
    )

    expected_len = n_img + seq_len
    assert inputs_embeds.shape == (1, expected_len, 1024), (
        f"Merged embeds shape mismatch: expected (1, {expected_len}, 1024), "
        f"got {inputs_embeds.shape}"
    )
    print(f"  inputs_embeds: {inputs_embeds.shape}  dtype={inputs_embeds.dtype}  "
          f"[verified: {n_img} img + {seq_len} lang = {expected_len}]")
    print(f"  attention_mask: {attention_mask.shape}")
    return inputs_embeds, attention_mask


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


def compute_heatmaps(all_attentions: tuple, n_img: int, seq_len: int) -> list:
    """
    Computes language→image attention heatmaps for each encoder layer.

    Skips index 0 (spatial_avg_pool global token) and uses indices 1..n_img-1
    as a 7×7 spatial grid. Returns 12 normalized (7, 7) numpy arrays.
    """
    import numpy as np

    n_spatial = n_img - 1
    grid_side = int(n_spatial ** 0.5)
    assert grid_side * grid_side == n_spatial, (
        f"Spatial token count {n_spatial} is not a perfect square — "
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


def save_visualizations(
    pil_img: Image.Image,
    heatmaps: list,
    all_attentions: tuple,
    n_img: int,
    args: argparse.Namespace,
) -> None:
    """Saves 4 visualization outputs to xai/outputs/."""
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    display_wh = (224, 224)
    display_hw = (224, 224)

    def upsample(m: np.ndarray) -> np.ndarray:
        return cv2.resize(m.astype(np.float32), display_wh, interpolation=cv2.INTER_CUBIC)

    def save_img_bgr(path: str, rgb_array: np.ndarray) -> None:
        cv2.imwrite(path, rgb_array[..., ::-1])

    out_dir = ensure_output_dir()

    fig, axes = plt.subplots(3, 4, figsize=(16, 12), facecolor="#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    for i, ax in enumerate(axes.flat):
        if i < len(heatmaps):
            overlay = overlay_heatmap(pil_img, upsample(heatmaps[i]), args.alpha, args.cmap)
            ax.imshow(overlay)
            ax.set_title(f"Layer {i}", fontsize=9, color="#e8e8e8", pad=3)
        ax.axis("off")
    fig.suptitle("P0-V: Language→Image Attention per Encoder Layer",
                 fontsize=13, color="#e8e8e8", y=1.01)
    plt.tight_layout()
    grid_path = os.path.join(out_dir, "p0v_per_layer_grid.png")
    plt.savefig(grid_path, bbox_inches="tight", facecolor="#1a1a2e", dpi=120)
    plt.close()
    print(f"  Saved: {grid_path}")

    layer_idx = min(args.layer, len(all_attentions) - 1)
    attn_layer = all_attentions[layer_idx][0]
    n_heads = attn_layer.shape[0]
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), facecolor="#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    for h_idx, ax in enumerate(axes.flat):
        if h_idx < n_heads:
            head_attn = attn_layer[h_idx, n_img:, 1:n_img]
            head_1d = head_attn.mean(dim=0).cpu().numpy()
            grid_side = int(head_1d.shape[0] ** 0.5)
            head_2d = head_1d.reshape(grid_side, grid_side)
            head_norm = (head_2d - head_2d.min()) / (head_2d.max() - head_2d.min() + 1e-8)
            overlay = overlay_heatmap(pil_img, upsample(head_norm), args.alpha, args.cmap)
            ax.imshow(overlay)
            ax.set_title(f"Head {h_idx}", fontsize=9, color="#e8e8e8", pad=3)
        ax.axis("off")
    fig.suptitle(f"P0-V: Per-Head Attention — Layer {layer_idx}",
                 fontsize=13, color="#e8e8e8", y=1.01)
    plt.tight_layout()
    head_path = os.path.join(out_dir, f"p0v_head_analysis_layer{layer_idx}.png")
    plt.savefig(head_path, bbox_inches="tight", facecolor="#1a1a2e", dpi=120)
    plt.close()
    print(f"  Saved: {head_path}")

    mean_h = np.mean(heatmaps, axis=0)
    mean_norm = (mean_h - mean_h.min()) / (mean_h.max() - mean_h.min() + 1e-8)
    agg_overlay = overlay_heatmap(pil_img, upsample(mean_norm), args.alpha, args.cmap)
    agg_path = os.path.join(out_dir, "p0v_aggregated.png")
    save_img_bgr(agg_path, agg_overlay)
    print(f"  Saved: {agg_path}")

    last_overlay = overlay_heatmap(pil_img, upsample(heatmaps[-1]), args.alpha, args.cmap)
    ov_path = os.path.join(out_dir, "p0v_overlay.png")
    save_img_bgr(ov_path, last_overlay)
    print(f"  Saved: {ov_path}")


def main(args: argparse.Namespace) -> None:
    import numpy as np

    start_time = time.time()
    ensure_output_dir()

    print("=" * 60)
    print("XVLA XAI — P0-V Raw Attention Map")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")

    print("\n[1] Loading model with encoder...")
    model = load_vision_tower(device, keep_language_encoder=True)
    encoder = model.language_model.model.encoder
    print(f"Model loaded. Encoder layers: {len(encoder.layers)}")
    report_vram(device, "after_load")

    if args.dry_run:
        print("\n[Dry run] Exiting early.")
        return

    print(f"\n[2] Loading image: {args.image}")
    if not os.path.isabs(args.image):
        search_paths = [
            args.image,
            os.path.join(OUTPUT_DIR, args.image),
            os.path.join(PROJECT_DIR, args.image),
        ]
        for p in search_paths:
            if os.path.exists(p):
                args.image = p
                break
        else:
            print(f"  Warning: Image not found: {args.image}")
            print("  Creating dummy 224x224 image for testing...")
            pil_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            args.image = "dummy_test.jpg"
    if os.path.exists(args.image):
        pil_img = load_image_pil(args.image)

    print("\n[3] Encoding image...")
    image_features, n_img = encode_image(pil_img, model, device)

    print("\n[4] Tokenizing instruction...")
    input_ids, seq_len = tokenize_instruction(args.instruction, device)

    print("\n[5] Building merged embeddings...")
    inputs_embeds, attention_mask = build_merged_embeds(
        model, image_features, input_ids, n_img, seq_len
    )

    print("\n[6] Extracting attention weights...")
    all_attentions = extract_attention_weights(encoder, inputs_embeds, attention_mask)

    seq_total = n_img + seq_len
    assert len(all_attentions) == 12, f"Expected 12 layers, got {len(all_attentions)}"
    expected_shape = (1, 16, seq_total, seq_total)
    assert all_attentions[0].shape == expected_shape, (
        f"Attention shape mismatch: expected {expected_shape}, got {all_attentions[0].shape}"
    )
    if torch.isnan(all_attentions[0]).any() or torch.isinf(all_attentions[0]).any():
        print("  [WARN] Attention contains NaN or Inf values!")
    else:
        print(f"  len(attentions): {len(all_attentions)}")
        print(f"  attentions[0] shape: {all_attentions[0].shape}")

    print("\n[7] Computing heatmaps...")
    heatmaps = compute_heatmaps(all_attentions, n_img, seq_len)
    print(f"  {len(heatmaps)} heatmaps computed, shape: {heatmaps[0].shape}")

    print("\n[8] Saving visualizations...")
    save_visualizations(pil_img, heatmaps, all_attentions, n_img, args)

    mean_h = np.mean(heatmaps, axis=0)
    p = mean_h / (mean_h.sum() + 1e-8)
    entropy = -np.sum(p * np.log(p + 1e-8))
    norm_entropy = entropy / np.log(mean_h.size)
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Attention entropy (normalized): {norm_entropy:.3f}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Outputs: xai/outputs/")
    print("=" * 60)


if __name__ == "__main__":
    _INSTR_SEEDS = "Pour coffee from the orange cup into the light blue cup."
    _INSTR_COFFEE = (
        "Pour coffee from the orange cup into the cup "
        "with the black-bordered letter D sticker."
    )

    parser = argparse.ArgumentParser(
        description="P0-V Raw Attention Map Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Training instructions for this model:\n"
            f"  seeds:  {_INSTR_SEEDS}\n"
            f"  coffee: {_INSTR_COFFEE}\n"
        ),
    )
    parser.add_argument(
        "--image",
        default="/home/tunx/XVLA/test_image/000625.png",
        help="Input image path (absolute, or relative to PROJECT_DIR / xai/)",
    )
    parser.add_argument(
        "--instruction",
        default=_INSTR_SEEDS,
        help=(
            "Instruction text fed to the language encoder. "
            "Use the actual training instructions for meaningful results "
            "(see --help epilog). Default: sunflower-seeds instruction."
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=11,
        help="Encoder layer index (0–11) for per-head analysis (default: 11 = last layer)",
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
        help="Matplotlib colormap name for heatmaps (default: turbo)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model and print shapes only — skip all visualization",
    )
    args = parser.parse_args()
    main(args)