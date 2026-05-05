#!/usr/bin/env python3
"""
XAI Phase 1a — Feature Map Visualization for DaViT Vision Encoder.

Runs a single forward pass through the XVLA model's DaViT vision tower
and visualizes mean-channel activation maps at each of the 4 stages.

DaViT spatial resolution pipeline:
  Input: 224×224
  Stage 0 (ConvEmbed+Block): 56×56 × 256 ch
  Stage 1 (ConvEmbed+Block): 28×28 × 512 ch
  Stage 2 (ConvEmbed+Block): 14×14 × 1024 ch  ← 9 attention blocks, most expressive
  Stage 3 (ConvEmbed+Block):  7×7  × 2048 ch  ← highest abstraction

Usage (on Linux server from xai/ directory):
    python3 xai_feature_maps.py [--image PATH] [--alpha 0.6] [--cmap turbo]

Outputs (in xai/outputs/):
    feature_maps_grid.png        — 2×5 grid: original + 4 stages (raw + overlay)
    feature_maps_combined.png    — average heatmap across all stages, overlaid
    feature_maps_stage{0-3}.png  — individual per-stage overlays
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xai_utils  # noqa: E402

from xai_utils import (
    PROJECT_DIR, OUTPUT_DIR,
    load_vision_tower, load_image_pil, pil_to_tensor,
    report_vram, ensure_output_dir,
    DAVIT_INPUT_SIZE,
)

STAGES = [
    {"idx": 0, "name": "Stage 0", "res": "56×56",  "ch": 256,  "depth": 1,
     "desc": "Low-level edges & textures"},
    {"idx": 1, "name": "Stage 1", "res": "28×28",  "ch": 512,  "depth": 1,
     "desc": "Mid-level shapes & contours"},
    {"idx": 2, "name": "Stage 2", "res": "14×14",  "ch": 1024, "depth": 9,
     "desc": "High-level semantics (9 attention blocks)"},
    {"idx": 3, "name": "Stage 3", "res": "7×7",    "ch": 2048, "depth": 1,
     "desc": "Global context & abstraction"},
]


class FeatureMapHook:
    """Captures block-output feature maps via a PyTorch forward hook."""

    def __init__(self, stage_idx: int):
        self.stage_idx = stage_idx
        self.features: torch.Tensor | None = None
        self._handle = None

    def attach(self, module: torch.nn.Module) -> "FeatureMapHook":
        self._handle = module.register_forward_hook(self._hook_fn)
        return self

    def _hook_fn(self, module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        self.features = out.detach().cpu().float()

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @property
    def spatial_size(self) -> int:
        """Side length of the square spatial grid (e.g. 56 for stage 0)."""
        return [56, 28, 14, 7][self.stage_idx]


AGGREGATION_MODES = {
    "centered_norm": "L2-norm of spatially-centered features (default, removes DC bias)",
    "norm":          "Raw L2-norm across channels",
    "mean":          "Mean across channels (fast but shows window attention artifacts)",
    "max":           "Max across channels (emphasizes strongest activations)",
}


def _aggregate_channels(feat_hwc: torch.Tensor, mode: str) -> np.ndarray:
    """
    Aggregates a (H, W, C) feature tensor into a (H, W) activation scalar map.

    'centered_norm' subtracts the per-channel spatial mean to remove the
    positional DC bias introduced by DaViT's window attention mechanism.
    """
    if mode == "centered_norm":
        centered = feat_hwc - feat_hwc.mean(dim=[0, 1], keepdim=True)
        return centered.norm(dim=-1).numpy()
    elif mode == "norm":
        return feat_hwc.norm(dim=-1).numpy()
    elif mode == "max":
        return feat_hwc.max(dim=-1).values.numpy()
    else:
        return feat_hwc.mean(dim=-1).numpy()


def extract_feature_maps(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    mode: str = "centered_norm",
) -> list[np.ndarray]:
    """
    Runs a forward pass and returns activation maps (one per DaViT stage),
    each normalized to [0, 1] and shaped (H, W).
    """
    hooks = [FeatureMapHook(i).attach(model.vision_tower.blocks[i])
             for i in range(len(STAGES))]

    with torch.no_grad():
        model.vision_tower.forward_features_unpool(image_tensor)

    maps = []
    for hook in hooks:
        feat = hook.features
        n = feat.shape[1]
        s = int(n ** 0.5)
        feat = feat[0].reshape(s, s, -1).float()

        act = _aggregate_channels(feat, mode)

        lo, hi = np.percentile(act, 2), np.percentile(act, 98)
        act = np.clip((act - lo) / (hi - lo + 1e-8), 0.0, 1.0)

        maps.append(act)
        hook.remove()

    return maps


def upsample_map(
    act_map: np.ndarray,
    target_hw: tuple[int, int],
    stage_idx: int = -1,
) -> np.ndarray:
    """
    Upscales an activation map to target_hw via bicubic interpolation.
    Applies per-stage Gaussian blur to suppress blocky upscaling artifacts.
    """
    h, w = target_hw
    up = cv2.resize(act_map.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    sigmas = {0: 0.015, 1: 0.02, 2: 0.05, 3: 0.04}
    sigma = h * sigmas.get(stage_idx, 0.025)
    up = cv2.GaussianBlur(up, (0, 0), sigmaX=sigma)
    return np.clip(up, 0.0, 1.0)


def apply_colormap(act_map: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    """Converts a [0,1] grayscale map → RGBA uint8 via a matplotlib colormap."""
    cmap = cm.get_cmap(cmap_name)
    return (cmap(act_map) * 255).astype(np.uint8)


def pil_letterbox(pil_img: Image.Image, height: int, width: int) -> Image.Image:
    """
    Applies the same letterboxing as resize_with_pad() to a PIL image.
    Returns a (width × height) image with black padding on left/top.
    """
    ih, iw = pil_img.height, pil_img.width
    ratio = max(iw / width, ih / height)
    rw = int(iw / ratio)
    rh = int(ih / ratio)
    resized = pil_img.resize((rw, rh), Image.BILINEAR)
    pw = max(0, width - rw)
    ph = max(0, height - rh)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(resized, (pw, ph))
    return canvas


def deletterbox_map(
    act_map: np.ndarray,
    pil_img: Image.Image,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """
    Removes letterbox padding from a feature map and stretches the content
    region to target_hw. Returns a [0,1] float32 array of shape target_hw.
    """
    H, W = act_map.shape
    ih, iw = pil_img.height, pil_img.width

    ratio = max(iw / W, ih / H)
    rw = int(iw / ratio)
    rh = int(ih / ratio)
    pw = max(0, W - rw)
    ph = max(0, H - rh)

    act_content = act_map[ph: ph + rh, pw: pw + rw]

    out_h, out_w = target_hw
    result = cv2.resize(
        act_content.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_CUBIC
    )
    return np.clip(result, 0.0, 1.0)


def overlay_heatmap(
    pil_img: Image.Image,
    act_map: np.ndarray,
    alpha: float = 0.55,
    cmap_name: str = "turbo",
) -> np.ndarray:
    """
    Overlays a heatmap on the original image with pixel-perfect spatial alignment.

    Crops the content region from the letterboxed feature map and stretches it
    to fill the display dimensions before blending with the original image.
    Returns an RGB uint8 numpy array (H, W, 3).
    """
    H, W = act_map.shape
    ih, iw = pil_img.height, pil_img.width

    ratio = max(iw / W, ih / H)
    rw = int(iw / ratio)
    rh = int(ih / ratio)
    pw = max(0, W - rw)
    ph = max(0, H - rh)

    act_content = act_map[ph: ph + rh, pw: pw + rw]

    act_display = cv2.resize(
        act_content.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC
    )
    act_display = np.clip(act_display, 0.0, 1.0)

    img_np = np.array(pil_img.resize((W, H), Image.BILINEAR)).astype(np.float32)

    heat_rgb = apply_colormap(act_display, cmap_name).astype(np.float32)[..., :3]
    blended = (1 - alpha) * img_np + alpha * heat_rgb
    return np.clip(blended, 0, 255).astype(np.uint8)


TITLE_FONT = {"fontsize": 9, "fontweight": "bold", "color": "#e8e8e8"}
LABEL_FONT = {"fontsize": 7.5, "color": "#b0b0b0"}


def _style_ax(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    ax.set_title(title, **TITLE_FONT, pad=4)
    if subtitle:
        ax.text(0.5, -0.06, subtitle, transform=ax.transAxes,
                ha="center", **LABEL_FONT)
    ax.axis("off")


def save_grid(
    pil_orig: Image.Image,
    activation_maps: list[np.ndarray],
    out_path: str,
    alpha: float,
    cmap_name: str,
    image_name: str,
) -> None:
    """
    Saves a 2×5 grid figure:
      Row 0: original | raw heatmaps (stage 0-3)
      Row 1: original | overlaid heatmaps (stage 0-3)
    """
    display_hw = DAVIT_INPUT_SIZE

    maps_up = [upsample_map(m, display_hw, i) for i, m in enumerate(activation_maps)]
    maps_dl = [deletterbox_map(m, pil_orig, display_hw) for m in maps_up]
    maps_colored = [apply_colormap(m, cmap_name) for m in maps_dl]
    maps_overlay = [overlay_heatmap(pil_orig, m, alpha, cmap_name) for m in maps_up]
    orig_resized = np.array(pil_orig.resize((display_hw[1], display_hw[0]), Image.BILINEAR))

    fig, axes = plt.subplots(
        2, 5,
        figsize=(18, 7.5),
        facecolor="#1a1a2e",
        gridspec_kw={"wspace": 0.06, "hspace": 0.22},
    )
    fig.patch.set_facecolor("#1a1a2e")

    _style_ax(axes[0, 0], "Original Image", image_name)
    axes[0, 0].imshow(orig_resized)

    for col, (stage, m_col) in enumerate(zip(STAGES, maps_colored), start=1):
        _style_ax(axes[0, col], f"{stage['name']}  {stage['res']}×{stage['ch']}ch", stage["desc"])
        axes[0, col].imshow(m_col[..., :3])

    _style_ax(axes[1, 0], "Original Image", "")
    axes[1, 0].imshow(orig_resized)

    for col, (stage, m_ov) in enumerate(zip(STAGES, maps_overlay), start=1):
        _style_ax(axes[1, col], f"{stage['name']}  (overlay)", "")
        axes[1, col].imshow(m_ov)

    for row_idx, label in enumerate(["Raw activation", "Overlay on image"]):
        axes[row_idx, 0].text(
            -0.08, 0.5, label, transform=axes[row_idx, 0].transAxes,
            fontsize=9, color="#9090b0", rotation=90,
            va="center", ha="center",
        )

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.012, pad=0.01, shrink=0.7)
    cbar.set_label("Activation intensity", color="#b0b0b0", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#b0b0b0")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#b0b0b0", fontsize=7)
    cbar.outline.set_edgecolor("#404060")

    fig.suptitle(
        "DaViT Vision Encoder — Mean-Channel Feature Maps\n"
        "XVLA Fine-tuned Model (xvla-pouring-0.1)",
        fontsize=12, color="#d0d0f0", y=1.01,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_combined(
    pil_orig: Image.Image,
    activation_maps: list[np.ndarray],
    out_path: str,
    alpha: float,
    cmap_name: str,
    image_name: str,
) -> None:
    """Saves the averaged heatmap across all 4 stages overlaid on the original image."""
    display_hw = DAVIT_INPUT_SIZE

    maps_up = [upsample_map(m, display_hw, i) for i, m in enumerate(activation_maps)]
    combined = np.mean(np.stack(maps_up, axis=0), axis=0)
    lo, hi = combined.min(), combined.max()
    combined = (combined - lo) / (hi - lo + 1e-8)

    combined_dl = deletterbox_map(combined, pil_orig, display_hw)
    overlay = overlay_heatmap(pil_orig, combined, alpha, cmap_name)
    raw_col = apply_colormap(combined_dl, cmap_name)
    orig_np = np.array(pil_orig.resize((display_hw[1], display_hw[0]), Image.BILINEAR))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor="#1a1a2e",
                              gridspec_kw={"wspace": 0.06})
    fig.patch.set_facecolor("#1a1a2e")

    _style_ax(axes[0], "Original Image", image_name)
    axes[0].imshow(orig_np)

    _style_ax(axes[1], "Combined Activation (all stages)", "Mean across Stage 0-3")
    axes[1].imshow(raw_col[..., :3])

    _style_ax(axes[2], "Combined Overlay", "High activation = model focus area")
    axes[2].imshow(overlay)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Activation", color="#b0b0b0", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#b0b0b0")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#b0b0b0", fontsize=7)
    cbar.outline.set_edgecolor("#404060")

    fig.suptitle(
        "DaViT — Combined Attention Heatmap (All Stages)\n"
        "Bright regions = where the vision encoder focuses",
        fontsize=11, color="#d0d0f0",
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_individual_stages(
    pil_orig: Image.Image,
    activation_maps: list[np.ndarray],
    out_dir: str,
    alpha: float,
    cmap_name: str,
) -> None:
    """Saves one overlay PNG per DaViT stage."""
    display_hw = DAVIT_INPUT_SIZE
    orig_np = np.array(pil_orig.resize((display_hw[1], display_hw[0]), Image.BILINEAR))

    for stage, act in zip(STAGES, activation_maps):
        up = upsample_map(act, display_hw, stage["idx"])
        overlay = overlay_heatmap(pil_orig, up, alpha, cmap_name)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor="#1a1a2e",
                                  gridspec_kw={"wspace": 0.05})
        fig.patch.set_facecolor("#1a1a2e")

        _style_ax(axes[0], "Original Image")
        axes[0].imshow(orig_np)

        _style_ax(
            axes[1],
            f"{stage['name']}  {stage['res']}×{stage['ch']}ch  (depth={stage['depth']})",
            stage["desc"],
        )
        axes[1].imshow(overlay)

        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("Activation", color="#b0b0b0", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="#b0b0b0")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#b0b0b0", fontsize=7)
        cbar.outline.set_edgecolor("#404060")

        out_path = os.path.join(out_dir, f"feature_maps_stage{stage['idx']}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize DaViT feature maps for XAI analysis.")
    p.add_argument(
        "--image",
        default="/home/tunx/XVLA/test_image/000665.png",
        help="Path to the input image (default: test_image.jpg in project root)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Heatmap overlay transparency — 0=original only, 1=heatmap only (default: 0.55)",
    )
    p.add_argument(
        "--mode",
        default="centered_norm",
        choices=list(AGGREGATION_MODES.keys()),
        help=(
            "Channel aggregation method.  'centered_norm' (default) removes\n"
            "positional DC bias from Window Attention.\n"
            + "\n".join(f"  {k}: {v}" for k, v in AGGREGATION_MODES.items())
        ),
    )
    p.add_argument(
        "--cmap",
        default="turbo",
        help="Matplotlib colormap name (default: turbo). Try: jet, inferno, plasma",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("XVLA XAI — DaViT Feature Map Visualization")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available.")
        return 1

    device = torch.device("cuda")
    print(f"\nDevice : {torch.cuda.get_device_name(device)}")
    report_vram(device, "before load")

    print("\nLoading vision tower …")
    t0 = time.perf_counter()
    model = load_vision_tower(device)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")
    report_vram(device, "after load")

    image_path = args.image
    if not os.path.exists(image_path):
        print(f"\n[WARN] Image not found: {image_path}")
        print("       Using a random synthetic image instead.")
        pil_orig = Image.new("RGB", (256, 256), color=(128, 64, 32))
        image_tensor = torch.rand(1, 3, 224, 224, device=device)
        image_name = "synthetic"
    else:
        print(f"\nImage: {image_path}")
        pil_orig = load_image_pil(image_path)
        image_tensor = pil_to_tensor(pil_orig, device)
        image_name = os.path.basename(image_path)

    print(f"Tensor: {image_tensor.shape}  range=[{image_tensor.min():.2f}, {image_tensor.max():.2f}]")

    print(f"\nExtracting feature maps [mode={args.mode}] via forward hooks …")
    t1 = time.perf_counter()
    maps = extract_feature_maps(model, image_tensor, mode=args.mode)
    torch.cuda.synchronize(device)
    print(f"Forward pass completed in {(time.perf_counter() - t1)*1000:.1f} ms")

    for stage, m in zip(STAGES, maps):
        print(f"  {stage['name']:8s}  {stage['res']:5s}  "
              f"act range=[{m.min():.3f}, {m.max():.3f}]  "
              f"mean={m.mean():.3f}")

    out_dir = ensure_output_dir()
    print(f"\nSaving visualizations to: {out_dir}")

    save_grid(
        pil_orig, maps,
        os.path.join(out_dir, "feature_maps_grid.png"),
        args.alpha, args.cmap, image_name,
    )
    save_combined(
        pil_orig, maps,
        os.path.join(out_dir, "feature_maps_combined.png"),
        args.alpha, args.cmap, image_name,
    )
    save_individual_stages(pil_orig, maps, out_dir, args.alpha, args.cmap)

    print("\n" + "=" * 60)
    print("DONE — Feature maps saved to xai/outputs/")
    print("Files generated:")
    print("  feature_maps_grid.png      — full 2×5 grid (raw + overlay)")
    print("  feature_maps_combined.png  — averaged across all stages")
    print("  feature_maps_stage{0-3}.png — individual per-stage overlays")
    print("\nNext step: inspect heatmaps — does the model focus on the cup")
    print("or on background regions (table color, lighting)?")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
