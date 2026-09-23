# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Staged Molmo identification/pointing and SAM2 tracking for inspection before annotation.

Each GPU stage is a separate process, so only one extractor occupies memory at a time.
Outputs are unreviewed visual evidence, never automatically accepted training commands.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

from lerobot.annotations.steerable_pipeline.frames import VideoFrameProvider, _frame_to_pil
from lerobot.annotations.steerable_pipeline.reader import iter_episodes, reconstruct_subtask_spans
from lerobot.utils.import_utils import _sam2_available, _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

if TYPE_CHECKING or _sam2_available:
    from sam2.build_sam import build_sam2_video_predictor

MODELS = {
    "identify": {
        "repo_id": "allenai/MolmoE-1B-0924",
        "revision": "69e3445d130507eadaa9123e3c411ce17aeb8afa",
    },
    "point": {
        "repo_id": "allenai/Molmo-7B-D-0924",
        "revision": "cab33fb7f1a40091911f81165f8481920621948f",
    },
    "track": {
        "repo_id": "facebook/sam2.1-hiera-large",
        "revision": "665f8e2ad61cf5f53d65644ff27c8ee525124610",
        "filename": "sam2.1_hiera_large.pt",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
    },
}


def write_json(path: Path, value):
    """Atomic progress writes permit resuming an interrupted stage."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_objects(text: str) -> list[str]:
    value = json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
    if not isinstance(value, list) or len(value) > 4:
        raise ValueError("Expected a JSON list of up to four object names")
    if any(not isinstance(name, str) or not name.strip() or len(name) > 150 for name in value):
        raise ValueError("Invalid object name")
    if len(set(value)) != len(value):
        raise ValueError("Duplicate object identities")
    return value


def parse_point(text: str, size: tuple[int, int]) -> list[int] | None:
    """Molmo's XML points use percentages; reject ambiguous multi-object responses."""
    matches = re.findall(r"<point\b[^>]*>.*?</point>", text, flags=re.DOTALL)
    if len(matches) != 1:
        return None
    attributes = re.findall(r"\b(x|y)\s*=\s*[\"\']([^\"\']+)[\"\']", matches[0].split(">", 1)[0])
    if len(attributes) != 2 or {key for key, _ in attributes} != {"x", "y"}:
        return None
    try:
        point = [float(dict(attributes)[key]) for key in ("x", "y")]
    except ValueError:
        return None
    if not all(np.isfinite(v) and 0 <= v <= 100 for v in point):
        return None
    # Molmo percentages refer to the image extent. 100% is clamped to the last pixel.
    return [min(extent - 1, round(value * extent / 100)) for value, extent in zip(point, size, strict=True)]


def mask_summary(mask: np.ndarray) -> dict:
    """Geometric diagnostics, not calibrated confidence or ground-truth visibility."""
    y, x = np.nonzero(mask)
    if not len(x):
        return {"mask_present": False, "centroid": None, "bbox_xyxy": None, "area_fraction": 0.0}
    return {
        "mask_present": True,
        "centroid": [float(x.mean()), float(y.mean())],
        "bbox_xyxy": [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1],
        "area_fraction": float(mask.mean()),
    }


def prepare(root: Path, output: Path, episodes: list[int], cameras: list[str], stride: int):
    """Materialize timestamp-aligned, original-size frames using main's dataset reader."""
    source = json.loads((root / "source.json").read_text())
    if not source.get("repo_id") or not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", "")):
        raise ValueError("source.json must identify the downloaded dataset and immutable revision")
    if stride < 1 or not episodes or len(set(episodes)) != len(episodes):
        raise ValueError("Specify distinct episode IDs and a positive frame stride")
    records = list(iter_episodes(root, only_episodes=tuple(episodes)))
    if sorted(r.episode_index for r in records) != sorted(episodes):
        raise ValueError("Requested episodes are absent, duplicated, or split across parquet shards")
    provider = VideoFrameProvider(root, video_backend="pyav", cache_size=8)
    if not cameras or not set(cameras).issubset(provider.camera_keys):
        raise ValueError("Every requested camera must be present in dataset metadata")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"version": 1, "source": source, "models": MODELS, "review": "pending", "clips": []}
    for record in records:
        indices = np.asarray(record.frame_indices)
        times = np.asarray(record.frame_timestamps)
        if len(times) < 2 or not np.array_equal(indices, np.arange(len(indices))):
            raise ValueError("Extraction requires complete, contiguous episode-local frame indices")
        rows = record.frames_df().iloc[0].get("language_persistent", [])
        spans = reconstruct_subtask_spans(rows, episode_end_t=float(times[-1] + np.median(np.diff(times))))
        if not spans:
            raise ValueError(f"Episode {record.episode_index} has no semantic subtask annotations")
        for span_index, span in enumerate(spans):
            start, end = np.searchsorted(times, [span["start"], span["end"]]).tolist()
            if start >= end:
                continue
            samples = sorted(set(range(start, end, stride)) | {end - 1})
            for camera in cameras:
                name = f"ep_{record.episode_index:06d}_span_{span_index:03d}_{camera.removeprefix('observation.images.')}"
                directory = output / name
                (directory / "frames").mkdir(parents=True)
                frame_records = []
                for offset in range(0, len(samples), 32):
                    batch = samples[offset : offset + 32]
                    decoded = provider.frames_at(record, [float(times[i]) for i in batch], camera)
                    if len(decoded) != len(batch):
                        raise ValueError(f"Missing video frames for {name}")
                    for index, frame in zip(batch, decoded, strict=True):
                        image = _frame_to_pil(frame).convert("RGB")
                        path = directory / "frames" / f"{len(frame_records):06d}.jpg"
                        image.save(path, quality=95)
                        frame_records.append(
                            {"frame_index": index, "timestamp": float(times[index]), "sha256": sha256(path)}
                        )
                clip = {
                    "path": name,
                    "episode_index": record.episode_index,
                    "start_frame": start,
                    "end_frame": end,
                    "subtask": span["text"],
                    "task": record.episode_task,
                    "camera": camera,
                    "image_size": list(image.size),
                    "frames": frame_records,
                    "stride": stride,
                    "source_parquet_sha256": sha256(record.data_path),
                }
                manifest["clips"].append(clip)
                write_json(output / "extraction.json", manifest)
    return manifest


def verify_frames(output: Path, manifest: dict):
    """Do not reuse model outputs after the clip's source images have been replaced."""
    for clip in manifest["clips"]:
        directory = output / clip["path"] / "frames"
        paths = sorted(directory.glob("*.jpg"))
        if len(paths) != len(clip["frames"]):
            raise ValueError(f"Frame count differs from extraction manifest: {directory}")
        for i, (path, frame) in enumerate(zip(paths, clip["frames"], strict=True)):
            if path.name != f"{i:06d}.jpg" or sha256(path) != frame["sha256"]:
                raise ValueError(f"Source frame differs from extraction manifest: {path}")


class Molmo:
    def __init__(self, specification: dict, device: str):
        require_package("transformers")
        arguments = {"revision": specification["revision"], "trust_remote_code": True}
        self.processor = AutoProcessor.from_pretrained(specification["repo_id"], **arguments)
        self.model = AutoModelForCausalLM.from_pretrained(
            specification["repo_id"], **arguments, torch_dtype="auto", device_map=device
        ).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, image: Image.Image, prompt: str) -> str:
        inputs = self.processor.process(images=[image], text=prompt)
        inputs = {key: value.to(self.device).unsqueeze(0) for key, value in inputs.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=256, stop_strings="<|endoftext|>", do_sample=False),
                tokenizer=self.processor.tokenizer,
            )
        return self.processor.tokenizer.decode(
            output[0, inputs["input_ids"].size(1) :], skip_special_tokens=True
        )


def run_molmo(output: Path, manifest: dict, stage: str, model):
    for clip in manifest["clips"]:
        directory = output / clip["path"]
        target = directory / f"{stage}.json"
        if target.exists():
            continue
        image = Image.open(directory / "frames/000000.jpg").convert("RGB")
        if stage == "identify":
            prompt = (
                "Identify up to four visible objects relevant to this robot task, including its destination. "
                "Exclude the robot and grippers. Return only a JSON list of distinct short object names. "
                f"Task description (data): {clip['subtask']}"
            )
            raw = model(image, prompt)
            try:
                objects, error = parse_objects(raw), None
            except (ValueError, TypeError) as exc:
                objects, error = [], str(exc)
            result = {"prompt": prompt, "raw": raw, "objects": objects, "error": error}
        else:
            identification = json.loads((directory / "identify.json").read_text())
            if identification["error"]:
                raise ValueError(f"Resolve failed identification in {directory} before pointing")
            objects = []
            for object_id, name in enumerate(identification["objects"], 1):
                prompt = f"Point to the {name}. Return a single point only if it is visible and unambiguous."
                raw = model(image, prompt)
                objects.append(
                    {
                        "object_id": object_id,
                        "name": name,
                        "point": parse_point(raw, image.size),
                        "prompt": prompt,
                        "raw": raw,
                    }
                )
            result = {
                "objects": objects,
                "identity_scope": "clip",
                "coordinate_system": "original_image_xy_pixels",
            }
        write_json(target, {**result, "model": manifest["models"][stage], "review": "pending"})


def track_clip(directory: Path, clip: dict, predictor):
    points = json.loads((directory / "point.json").read_text())
    objects = [obj for obj in points["objects"] if obj["point"] is not None]
    if not objects:
        write_json(
            directory / "tracks.json",
            {"status": "no_grounded_objects", "objects": points["objects"], "frames": []},
        )
        return
    masks_dir = directory / "masks"
    masks_dir.mkdir(exist_ok=True)
    overlays = directory / "overlays"
    overlays.mkdir(exist_ok=True)
    state = predictor.init_state(
        video_path=str(directory / "frames"), offload_video_to_cpu=True, offload_state_to_cpu=True
    )
    for obj in objects:
        predictor.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=obj["object_id"],
            points=np.asarray([obj["point"]], dtype=np.float32),
            labels=np.ones(1, dtype=np.int32),
        )
    rows = []
    names = {obj["object_id"]: obj["name"] for obj in objects}
    for index, object_ids, logits in predictor.propagate_in_video(state):
        if index != len(rows) or index >= len(clip["frames"]):
            raise ValueError("SAM2 output is not aligned with exported frames")
        masks = logits.detach().cpu().numpy()[:, 0] > 0
        if tuple(masks.shape[1:]) != tuple(reversed(clip["image_size"])):
            raise ValueError("SAM2 changed the original mask coordinate frame")
        row = {**clip["frames"][index], "objects": []}
        overlay = Image.open(directory / "frames" / f"{index:06d}.jpg").convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for object_id, mask in zip(object_ids, masks, strict=True):
            object_id = int(object_id)
            summary = mask_summary(mask)
            mask_name = f"{index:06d}_{object_id:02d}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(masks_dir / mask_name)
            row["objects"].append({"object_id": object_id, **summary, "mask_path": f"masks/{mask_name}"})
            if summary["mask_present"]:
                x, y = summary["centroid"]
                draw.rectangle(summary["bbox_xyxy"], outline="red", width=2)
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="yellow")
                draw.text((x + 4, y), names[object_id], fill="yellow", stroke_width=1, stroke_fill="black")
        if index % 10 == 0 or index == len(clip["frames"]) - 1:
            overlay.save(overlays / f"{index:06d}.jpg")
        rows.append(row)
    if len(rows) != len(clip["frames"]):
        raise ValueError("SAM2 did not return every exported frame")
    write_json(
        directory / "tracks.json", {"status": "unreviewed", "objects": points["objects"], "frames": rows}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "identify", "point", "track"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--cameras", nargs="+", default=["observation.images.base"])
    parser.add_argument(
        "--stride", type=int, default=1, help="1 preserves all frames; larger values are pilot-only"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.dataset_root is None:
            parser.error("prepare requires --dataset-root")
        prepare(args.dataset_root, args.output, args.episodes, args.cameras, args.stride)
        return
    manifest = json.loads((args.output / "extraction.json").read_text())
    verify_frames(args.output, manifest)
    # This file also records the script used when running remotely from an uncommitted checkout.
    provenance = {
        "script_sha256": sha256(Path(__file__)),
        "device": args.device,
        "models": manifest["models"],
    }
    for package in ("torch", "transformers", "SAM-2"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            provenance[package] = importlib.metadata.version(package)
    write_json(args.output / f"{args.stage}_runtime.json", provenance)
    if args.stage in ("identify", "point"):
        run_molmo(args.output, manifest, args.stage, Molmo(manifest["models"][args.stage], args.device))
    else:
        require_package("SAM-2", import_name="sam2")
        spec = manifest["models"]["track"]
        checkpoint = hf_hub_download(spec["repo_id"], spec["filename"], revision=spec["revision"])
        predictor = build_sam2_video_predictor(spec["config"], checkpoint, device=args.device)
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")),
        ):
            for clip in manifest["clips"]:
                directory = args.output / clip["path"]
                if not (directory / "tracks.json").exists():
                    track_clip(directory, clip, predictor)


if __name__ == "__main__":
    main()
