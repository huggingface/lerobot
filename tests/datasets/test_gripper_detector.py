# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image


@pytest.fixture
def detector():
    return runpy.run_path(str(Path(__file__).parents[2] / "examples/rebot_agent/gripper_detector.py"))


def test_ambiguous_and_invalid_detections_stay_missing(detector):
    result = {
        "labels": torch.tensor([0, 0, 1]),
        "scores": torch.tensor([0.95, 0.94, 0.99]),
        "boxes": torch.tensor([[1.0, 2.0, 5.0, 6.0], [9.0, 9.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
    }
    predictions = detector["per_arm_predictions"](result, (16, 12))
    assert predictions["left_gripper"]["status"] == "ambiguous"
    assert predictions["right_gripper"]["status"] == "invalid"
    assert all(value["bbox_xyxy"] is None for value in predictions.values())
    result = {key: value[:1] for key, value in result.items()}
    predictions = detector["per_arm_predictions"](result, (16, 12))
    assert predictions["left_gripper"]["bbox_xyxy"] == [1.0, 2.0, 5.0, 6.0]
    assert predictions["right_gripper"]["status"] == "missing"


def test_detr_processor_backward_and_reload_with_two_arm_and_empty_targets(detector, tmp_path):
    transformers = pytest.importorskip("transformers")
    processor = transformers.DetrImageProcessor(size={"height": 64, "width": 64})
    image = Image.new("RGB", (16, 12))
    annotations = [
        {"image_id": 0, "category_id": i, "bbox": [1 + i * 8, 2, 4, 6], "area": 24, "iscrowd": 0}
        for i in range(2)
    ]
    batch, _ = detector["collate"](
        [
            (image, {"image_id": 0, "annotations": annotations}, {"id": 0}),
            (image, {"image_id": 1, "annotations": []}, {"id": 1}),
        ],
        processor,
    )
    assert batch["labels"][0]["class_labels"].tolist() == [0, 1]
    torch.testing.assert_close(batch["labels"][0]["boxes"][0], torch.tensor([3 / 16, 5 / 12, 4 / 16, 6 / 12]))
    assert len(batch["labels"][1]["class_labels"]) == 0
    backbone = transformers.ResNetConfig(
        embedding_size=8,
        hidden_sizes=[8, 16, 32, 64],
        depths=[1, 1, 1, 1],
        layer_type="basic",
        out_features=["stage4"],
    )
    config = transformers.DetrConfig(
        backbone_config=backbone,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        num_queries=4,
        id2label=detector["LABELS"],
        label2id={v: k for k, v in detector["LABELS"].items()},
    )
    model = transformers.DetrForObjectDetection(config)
    inputs = detector["to_device"](batch, "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    output = model(**inputs)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1, error_if_nonfinite=True)
    assert norm > 0
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(**inputs).logits
    model.save_pretrained(tmp_path)
    processor.save_pretrained(tmp_path)
    provenance = detector["checkpoint_provenance"](tmp_path)
    assert provenance["optimizer_restored"] is False
    assert provenance["checkpoint_hashes"] == {
        "model.safetensors": detector["file_hash"](tmp_path / "model.safetensors")
    }
    restored = transformers.DetrForObjectDetection.from_pretrained(tmp_path, local_files_only=True).eval()
    assert restored.config.id2label == detector["LABELS"]
    with torch.no_grad():
        torch.testing.assert_close(restored(**inputs).logits, expected)
    # Exercise the actual extraction handoff, including its completion manifest.
    visual = tmp_path / "visual"
    frames = visual / "clip/frames"
    frames.mkdir(parents=True)
    image.save(frames / "000000.jpg")
    frame = {"frame_index": 0, "timestamp": 0.0, "sha256": detector["file_hash"](frames / "000000.jpg")}
    manifest = {
        "source": {"repo_id": "fixture", "revision": "fixture"},
        "clips": [
            {
                "path": "clip",
                "episode_index": 0,
                "camera": "observation.images.base",
                "image_size": [16, 12],
                "frames": [frame],
            }
        ],
    }
    (visual / "extraction.json").write_text(json.dumps(manifest))
    output = tmp_path / "predictions"
    detector["extract"](
        SimpleNamespace(
            checkpoint=tmp_path, visual=visual, output=output, device="cpu", threshold=0.7, margin=0.1
        )
    )
    exported = json.loads((output / "gripper_manifest.json").read_text())
    assert exported["visual_manifest_sha256"] == detector["file_hash"](visual / "extraction.json")
    assert exported["clips"] == [{"path": "clip", "sha256": detector["file_hash"](output / "clip.json")}]
    assert exported["config_sha256"] == detector["file_hash"](tmp_path / "config.json")
    prediction = json.loads((output / "clip.json").read_text())
    assert {k: prediction["frames"][0][k] for k in frame} == frame
    assert set(prediction["frames"][0]["arms"]) == {"left_gripper", "right_gripper"}


def test_warm_start_rejects_generic_or_mislabeled_checkpoint(detector, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"id2label": {"0": "person", "1": "bicycle"}}))
    with pytest.raises(ValueError, match="per-arm ReBot"):
        detector["checkpoint_provenance"](tmp_path)
