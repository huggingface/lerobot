from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy

HERE = Path(__file__).resolve().parent
ART = Path("/home/ubuntu24/Teleop/artifacts")
TRAIN = ART / "training/task1_picklift_act_csource_crender_v1/full_200k"
CKPT = TRAIN / "checkpoints/200000/pretrained_model"
DATASET = ART / "datasets/task1_picklift_act_csource_crender_v1/combined48_v1"
EVIDENCE = ART / "evidence/task1_picklift_act_csource_crender_v1/training_result_v1"
LOG = ART / "evidence/task1_picklift_act_csource_crender_v1/training_v1/formal.log"
EXPECTED_STATS = "947d612d48280a98f3f6aeb37744e0aae9f8ea2034b6def24ebcf62c18ff4651"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def final_metric() -> dict[str, str]:
    lines = LOG.read_text(errors="replace").replace("\r", "\n").splitlines()
    line = [row for row in lines if re.search(r"ot_train\.py:\d+ step:", row)][-1]
    fields = dict(re.findall(r"([A-Za-z0-9_/]+):(\S+)", line))
    return {key: fields[key] for key in ("step", "loss", "l1_loss", "kld_loss", "grdn", "smp/s")}


def main() -> None:
    if EVIDENCE.exists():
        raise FileExistsError(EVIDENCE)
    if "End of training" not in LOG.read_text(errors="replace"):
        raise RuntimeError("formal 200k training incomplete")
    counts_path = TRAIN / "matched_two_stream_sampling_counts.json"
    counts = json.loads(counts_path.read_text())
    if counts["actual_samples_seen_by_main_process"] != {"real24": 800000, "source_b": 800000}:
        raise RuntimeError("formal sample counts mismatch")
    checkpoints = {
        str(step): sha(TRAIN / f"checkpoints/{step:06d}/pretrained_model/model.safetensors")
        for step in range(20000, 200001, 20000)
    }
    dataset = LeRobotDataset(
        "local/task1_picklift_real24_localsim24gap_rerender_additive_v1",
        root=DATASET,
        video_backend="pyav",
    )
    if dataset.meta.total_episodes != 48 or len(dataset) != 6196:
        raise RuntimeError("combined Dataset identity mismatch")
    model = ACTPolicy.from_pretrained(CKPT).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy_cfg=model.config,
        pretrained_path=str(CKPT),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    samples = []
    for index, domain in ((0, "real24"), (4263, "rerender_localsim_gap24")):
        sample = dataset[index]
        inputs = {
            "observation.state": sample["observation.state"].unsqueeze(0),
            "observation.images.front": sample["observation.images.front"].unsqueeze(0),
        }
        model.reset()
        with torch.inference_mode():
            action = post(model.select_action(pre(inputs))).cpu().numpy()
        if action.shape != (1, 6) or not np.isfinite(action).all():
            raise RuntimeError(f"invalid offline inference for {domain}")
        samples.append({"domain": domain, "index": index, "finite": True, "shape": [1, 6], "action": action[0].tolist()})
    stats = CKPT / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    if sha(stats) != EXPECTED_STATS:
        raise RuntimeError("Real24 processor/stats identity mismatch")
    tensors = load_file(stats)
    if not np.allclose(tensors["observation.images.front.mean"].numpy().reshape(-1), [.485, .456, .406], atol=1e-7):
        raise RuntimeError("ImageNet visual stats mismatch")
    validation = {
        "status": "pass",
        "cuda_reload": True,
        "checkpoint": str(CKPT),
        "samples": samples,
        "hardware_accessed": False,
    }
    model_row = {
        "model_id": "ACT_R24_LocalSim24Gap_CRender_200k",
        "checkpoint": str(CKPT),
        "model_sha256": sha(CKPT / "model.safetensors"),
        "config_sha256": sha(CKPT / "config.json"),
        "train_config_sha256": sha(CKPT / "train_config.json"),
        "policy_preprocessor_sha256": sha(CKPT / "policy_preprocessor.json"),
        "processor_stats_sha256": sha(stats),
    }
    result = {
        "schema": "task1_act_csource_crender_training_result_v1",
        "status": "offline_training_complete_ready_for_separately_authorized_real_eval",
        "selected_step": 200000,
        "model": model_row,
        "formal": {
            "metrics": final_metric(),
            "checkpoints": checkpoints,
            "sampling_counts": counts,
            "log_sha256": sha(LOG),
        },
        "offline_validation": validation,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "boundaries": {"hardware_accessed": False, "rollout_started": False, "paper_effect_claim": False},
    }
    write(EVIDENCE / "offline_validation.json", validation)
    write(EVIDENCE / "training_result_v1.json", result)
    write(HERE / "training_result_v1.json", result)
    files = [EVIDENCE / "offline_validation.json", EVIDENCE / "training_result_v1.json", HERE / "training_result_v1.json"]
    (EVIDENCE / "hashes.sha256").write_text("".join(f"{sha(path)}  {path}\n" for path in files))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
