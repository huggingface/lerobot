# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""One warm native hand/camera process per HF Job, with a private MANO overlay."""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
from pathlib import Path

from .assets import checkpoint_guard, file_checksum
from .config import RUNTIME_REVISION, HandBackendConfig


def provisioned_environment(config: HandBackendConfig, mano: dict, directory: Path) -> dict[str, str]:
    root = Path(config.root)
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("The HF Jobs hand image must contain git")
    head = subprocess.check_output([git, "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != RUNTIME_REVISION:
        raise ValueError(f"Hand runtime must be pinned to {RUNTIME_REVISION}")
    subprocess.run([git, "-C", str(root), "diff", "--quiet", "HEAD", "--"], check=True)
    if not re.fullmatch(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*", config.module):
        raise ValueError("Configure the installed hand runtime module in the operator image")
    for name in ("python", "macrodata_python", "droid_python", "droid_moge_checkpoint"):
        if not Path(getattr(config, name)).is_file():
            raise ValueError(f"The HF Jobs image is missing {name}")
    manifest = Path(config.assets_manifest)
    if (
        not re.fullmatch(r"[a-f0-9]{64}", config.assets_sha256)
        or file_checksum(manifest)[0] != config.assets_sha256
    ):
        raise ValueError("Worker asset-manifest digest differs from the pinned image")
    assets = json.loads(manifest.read_text())
    if not assets.get("files"):
        raise ValueError("The hand image must list its non-MANO model and runtime files")
    for asset in assets["files"]:
        if "mano" in Path(asset["path"]).name.lower():
            raise ValueError("The shared image must not contain MANO assets")
        if file_checksum(Path(asset["path"]))[0] != asset["sha256"]:
            raise ValueError("Provisioned hand runtime asset changed")
    environment = os.environ.copy()
    for name in ("HF_HOME", "TORCH_HOME"):
        value = assets.get("environment", {}).get(name)
        if not value or not Path(value).is_dir():
            raise ValueError(f"The hand image must provision {name}")
        environment[name] = value
    guard = checkpoint_guard(directory, mano)
    environment.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="0",
        PYTHONPATH=os.pathsep.join([str(guard), str(root / "src")]),
    )
    # HF authentication is used only by the controller's private asset pull.
    environment.pop("HF_TOKEN", None)
    environment.pop("ANNOTATION_MANIFEST", None)
    return environment


class HandAnnotator:
    def __init__(self, config: HandBackendConfig, mano: dict, work: Path):
        self.config, self.work = config, work
        work.mkdir(parents=True, exist_ok=True)
        environment = provisioned_environment(config, mano, work / "private-imports")
        self.log = (work / "native.log").open("w")
        self.process = subprocess.Popen(
            [
                config.python,
                str(Path(__file__).with_name("bridge.py")),
                json.dumps(config.__dict__),
                str(work),
            ],
            cwd=config.root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.responses: queue.Queue = queue.Queue()
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("The hand worker must expose its request and response pipes")
        self.lock = threading.Lock()

        def reader():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)

        threading.Thread(target=reader, daemon=True).start()

    def annotate(self, video: Path, calibration: dict, directory: Path) -> dict:
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError("The native hand worker exited; inspect the Job logs")
            request = {
                "video": str(video.resolve()),
                "calibration": calibration,
                "directory": str(directory.resolve()),
            }
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            try:
                line = self.responses.get(timeout=self.config.timeout_s)
            except queue.Empty:
                self.close()
                raise TimeoutError("Native hand annotation timed out") from None
            if line is None:
                raise RuntimeError("The native hand worker exited before returning an episode")
            result = json.loads(line)
            if result.get("status") != "ok":
                raise RuntimeError("Native hand annotation failed; inspect the Job's native log")
            return result["result"]

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self.log.close()
