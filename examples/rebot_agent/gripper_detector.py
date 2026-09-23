# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Fine-tune DETR on reviewed ReBot gripper boxes and extract per-arm visual evidence.

The COCO-pretrained classifier is replaced; only a trained two-arm checkpoint can
be used for extraction. Detection outputs still require visual review.
"""

import argparse
import hashlib
import importlib.metadata
import json
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoImageProcessor, DetrForObjectDetection

BASE_MODEL = "facebook/detr-resnet-50"
BASE_REVISION = "1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b"
LABELS = {0: "left_gripper", 1: "right_gripper"}


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class GripperDataset(Dataset):
    def __init__(self, path: Path, image_root: Path | None = None):
        self.data = json.loads(path.read_text())
        if {c["id"]: c["name"] for c in self.data["categories"]} != LABELS:
            raise ValueError("Training requires reviewed left_gripper/right_gripper labels")
        self.images = self.data["images"]
        self.root = image_root or Path(self.data["info"]["image_root"])
        self.annotations = {im["id"]: [] for im in self.images}
        for annotation in self.data["annotations"]:
            self.annotations[annotation["image_id"]].append(annotation)
        for im in self.images:
            if file_hash(self.root / im["file_name"]) != im["sha256"]:
                raise ValueError("Detector source image differs from reviewed export")
            ids = [a["category_id"] for a in self.annotations[im["id"]]]
            if any(c not in LABELS for c in ids) or len(ids) != len(set(ids)):
                raise ValueError("Each physical arm may have at most one box per image")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im = self.images[index]
        image = Image.open(self.root / im["file_name"]).convert("RGB")
        return image, {"image_id": im["id"], "annotations": self.annotations[im["id"]]}, im


def collate(items, processor):
    images, annotations, metadata = zip(*items, strict=True)
    batch = processor(images=list(images), annotations=list(annotations), return_tensors="pt")
    return batch, metadata


def to_device(batch, device):
    return {
        key: [{k: v.to(device) for k, v in target.items()} for target in value]
        if key == "labels"
        else value.to(device)
        for key, value in batch.items()
    }


def per_arm_predictions(result: dict, size: tuple[int, int], margin: float = 0.1) -> dict:
    """Keep ambiguous or absent grippers missing; never interpolate an occluded track."""
    output = {}
    for label, name in LABELS.items():
        indices = torch.where(result["labels"] == label)[0]
        indices = indices[torch.argsort(result["scores"][indices], descending=True)]
        if not len(indices):
            output[name] = {"status": "missing", "bbox_xyxy": None, "score": None}
            continue
        best = indices[0]
        score = float(result["scores"][best])
        if len(indices) > 1 and score - float(result["scores"][indices[1]]) < margin:
            output[name] = {"status": "ambiguous", "bbox_xyxy": None, "score": score}
            continue
        box = result["boxes"][best].detach().cpu().numpy().astype(float)
        box[[0, 2]] = np.clip(box[[0, 2]], 0, size[0])
        box[[1, 3]] = np.clip(box[[1, 3]], 0, size[1])
        if not np.isfinite(box).all() or box[0] >= box[2] or box[1] >= box[3]:
            output[name] = {"status": "invalid", "bbox_xyxy": None, "score": score}
        else:
            output[name] = {"status": "detected", "bbox_xyxy": box.tolist(), "score": score}
    return output


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    def area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    return intersection / (area(a) + area(b) - intersection)


@torch.inference_mode()
def evaluate(model, processor, loader, device, threshold, margin):
    model.eval()
    counts = {name: {"tp": 0, "fp": 0, "fn": 0, "center_errors_px": []} for name in LABELS.values()}
    losses = []
    for batch, metadata in loader:
        output = model(**to_device(batch, device))
        if not torch.isfinite(output.loss):
            raise ValueError("Non-finite detector validation loss")
        losses.append(float(output.loss))
        results = processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=[(im["height"], im["width"]) for im in metadata]
        )
        for result, im in zip(results, metadata, strict=True):
            predictions = per_arm_predictions(result, (im["width"], im["height"]), margin)
            targets = {a["category_id"]: a["bbox"] for a in loader.dataset.annotations[im["id"]]}
            for label, name in LABELS.items():
                pred = predictions[name]["bbox_xyxy"]
                target = targets.get(label)
                truth = (
                    [target[0], target[1], target[0] + target[2], target[1] + target[3]] if target else None
                )
                matched = pred is not None and truth is not None and iou(pred, truth) >= 0.5
                counts[name]["tp"] += int(matched)
                counts[name]["fp"] += int(pred is not None and not matched)
                counts[name]["fn"] += int(truth is not None and not matched)
                if matched:
                    delta = (np.asarray(pred[:2]) + pred[2:] - np.asarray(truth[:2]) - truth[2:]) / 2
                    counts[name]["center_errors_px"].append(float(np.linalg.norm(delta)))
    for values in counts.values():
        tp, fp, fn = (values[k] for k in ("tp", "fp", "fn"))
        values["precision"] = tp / (tp + fp) if tp + fp else None
        values["recall"] = tp / (tp + fn) if tp + fn else None
        errors = values.pop("center_errors_px")
        values["mean_center_error_px_on_matches"] = float(np.mean(errors)) if errors else None
    return {
        "loss": float(np.mean(losses)),
        "threshold": threshold,
        "ambiguity_margin": margin,
        "iou_threshold": 0.5,
        "arms": counts,
    }


def train(args):
    require_package("transformers", extra="transformers-dep")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Use positive epochs and batch size")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Run detector training inside the science-cluster GPU allocation")
    torch.manual_seed(42)
    train_set, validation = (
        GripperDataset(args.labels / f"{split}.json", args.image_root) for split in ("train", "validation")
    )
    if not len(train_set) or not len(validation):
        raise ValueError("Both detector splits must contain reviewed images")
    if {im["episode_index"] for im in train_set.images} & {im["episode_index"] for im in validation.images}:
        raise ValueError("Detector train/validation episodes overlap")
    if train_set.data["info"]["labels_sha256"] != validation.data["info"]["labels_sha256"]:
        raise ValueError("Detector splits came from different reviewed label sets")
    for filename, key in (("reviewed_labels.json", "labels_sha256"), ("source_pack.json", "pack_sha256")):
        if file_hash(args.labels / filename) != train_set.data["info"][key]:
            raise ValueError("Reviewed labels or source pack differ from the detector export")
    args.output.mkdir(parents=True, exist_ok=False)
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    model = DetrForObjectDetection.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        id2label=LABELS,
        label2id={v: k for k, v in LABELS.items()},
        ignore_mismatched_sizes=True,
    ).to(args.device)
    loaders = [
        DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle, collate_fn=partial(collate, processor=processor)
        )
        for ds, shuffle in ((train_set, True), (validation, False))
    ]
    params = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and "backbone" not in n],
            "lr": 1e-4,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and "backbone" in n],
            "lr": 1e-5,
        },
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
    provenance = {
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "labels_sha256": train_set.data["info"]["labels_sha256"],
        "pack_sha256": train_set.data["info"]["pack_sha256"],
        "script_sha256": file_hash(Path(__file__)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "smoke": args.smoke,
        "seed": 42,
        "image_root": str(train_set.root.resolve()),
        "torch": str(torch.__version__),
        "transformers": importlib.metadata.version("transformers"),
        "train_annotations_sha256": file_hash(args.labels / "train.json"),
        "validation_annotations_sha256": file_hash(args.labels / "validation.json"),
    }
    for name in ("train.json", "validation.json", "source_pack.json", "reviewed_labels.json", "report.json"):
        (args.output / name).write_bytes((args.labels / name).read_bytes())
    (args.output / "training.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for epoch in range(1 if args.smoke else args.epochs):
        model.train()
        losses = []
        for step, (batch, _) in enumerate(loaders[0]):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**to_device(batch, args.device)).loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite detector training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            if args.smoke and step == 1:
                break
        metrics = evaluate(model, processor, loaders[1], args.device, args.threshold, args.margin)
        metrics.update(epoch=epoch, training_loss=float(np.mean(losses)))
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
        checkpoint = args.output / "checkpoint"
        model.save_pretrained(checkpoint)
        processor.save_pretrained(checkpoint)
    # Exercise the saved artifact itself. This report is not a detector-quality approval.
    del model, optimizer, params, loss
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    reloaded = DetrForObjectDetection.from_pretrained(checkpoint).to(args.device)
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    reload_loader = DataLoader(
        validation, batch_size=args.batch_size, collate_fn=partial(collate, processor=processor)
    )
    metrics = evaluate(reloaded, processor, reload_loader, args.device, args.threshold, args.margin)
    (args.output / "reload_validation.json").write_text(json.dumps(metrics, indent=2) + "\n")


@torch.inference_mode()
def extract(args):
    require_package("transformers", extra="transformers-dep")
    model = DetrForObjectDetection.from_pretrained(args.checkpoint).to(args.device).eval()
    if model.config.id2label != LABELS:
        raise ValueError("Use a trained per-arm ReBot detector, not a generic COCO checkpoint")
    processor = AutoImageProcessor.from_pretrained(args.checkpoint)
    manifest = json.loads((args.visual / "extraction.json").read_text())
    weights = {path.name: file_hash(path) for path in args.checkpoint.glob("*.safetensors")}
    if not weights:
        raise ValueError("Checkpoint must contain safetensors weights for provenance")
    args.output.mkdir(parents=True, exist_ok=False)
    export_manifest = {
        "kind": "gripper_predictions",
        "source": manifest["source"],
        "visual_manifest_sha256": file_hash(args.visual / "extraction.json"),
        "checkpoint_hashes": weights,
        "config_sha256": file_hash(args.checkpoint / "config.json"),
        "extractor_sha256": file_hash(Path(__file__)),
        "review": "pending",
        "clips": [],
    }
    for clip in manifest["clips"]:
        rows = []
        overlay_dir = args.output / clip["path"]
        overlay_dir.mkdir()
        for i, frame in enumerate(clip["frames"]):
            path = args.visual / clip["path"] / "frames" / f"{i:06d}.jpg"
            if file_hash(path) != frame["sha256"]:
                raise ValueError("Source frame changed after visual extraction")
            image = Image.open(path).convert("RGB")
            output = model(**processor(images=image, return_tensors="pt").to(args.device))
            result = processor.post_process_object_detection(
                output, threshold=args.threshold, target_sizes=[(image.height, image.width)]
            )[0]
            predictions = per_arm_predictions(result, image.size, args.margin)
            row = {**frame, "arms": predictions}
            if i % 10 == 0 or i == len(clip["frames"]) - 1:
                overlay = image.copy()
                draw = ImageDraw.Draw(overlay)
                for name, prediction in predictions.items():
                    box = prediction["bbox_xyxy"]
                    if box is not None:
                        color = "cyan" if name == "left_gripper" else "magenta"
                        draw.rectangle(box, outline=color, width=2)
                        draw.text(
                            (box[0], max(0, box[1] - 14)),
                            f"{name} {prediction['score']:.2f}",
                            fill=color,
                            stroke_width=1,
                            stroke_fill="black",
                        )
                row["overlay_path"] = f"{clip['path']}/{i:06d}.jpg"
                overlay.save(args.output / row["overlay_path"])
            rows.append(row)
        result = {
            "source": manifest["source"],
            "episode_index": clip["episode_index"],
            "camera": clip["camera"],
            "image_size": clip["image_size"],
            "checkpoint_hashes": weights,
            "threshold": args.threshold,
            "ambiguity_margin": args.margin,
            "review": "pending",
            "frames": rows,
        }
        (args.output / f"{clip['path']}.json").write_text(json.dumps(result, indent=2) + "\n")
        export_manifest["clips"].append(
            {"path": clip["path"], "sha256": file_hash(args.output / f"{clip['path']}.json")}
        )
    # Only a complete extraction can be exported into the dataset's language events.
    (args.output / "gripper_manifest.json").write_text(json.dumps(export_manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "extract"])
    parser.add_argument("--labels", type=Path, help="Directory exported by gripper_labels.py")
    parser.add_argument("--image-root", type=Path, help="Label-pack directory after transfer to another host")
    parser.add_argument("--visual", type=Path, help="Prepared extract_visual.py output")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or not 0 <= args.margin < 1:
        parser.error("Use a score threshold in (0,1) and ambiguity margin in [0,1)")
    if args.command == "train":
        if args.labels is None:
            parser.error("train requires --labels")
        train(args)
    else:
        if args.visual is None or args.checkpoint is None:
            parser.error("extract requires --visual and --checkpoint")
        extract(args)


if __name__ == "__main__":
    main()
