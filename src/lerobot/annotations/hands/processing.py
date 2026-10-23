# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Bounded warm camera/hand processes owned by one annotation Job."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import threading
import uuid
from concurrent.futures import Future
from dataclasses import asdict
from pathlib import Path

from .assets import file_checksum, verify_assets
from .config import GEM_REVISION, RUNTIME_REVISION, HandBackendConfig


def provisioned_environment(config: HandBackendConfig) -> dict[str, str]:
    source = Path(config.root) / "runtime-source.json"
    if file_checksum(source)[0] != config.runtime_source_sha256:
        raise ValueError("Camera adapter source differs from the pinned image")
    receipt = json.loads(source.read_text())
    if receipt["revision"] != RUNTIME_REVISION:
        raise ValueError("The image contains a different camera adapter revision")
    for entry in receipt["files"]:
        if file_checksum(Path(config.root) / entry["path"]) != (entry["sha256"], entry["size"]):
            raise ValueError("Camera adapter source changed")
    git = shutil.which("git")
    if git is None:
        raise ValueError("The native image requires Git to verify its frozen source")
    for root, revision in ((config.gem_root, GEM_REVISION),):
        # The immutable image checkout belongs to root; Jobs run as UID 1000.
        # Trust only this configured checkout for these read-only checks.
        command = [git, "-c", f"safe.directory={Path(root).resolve()}", "-C", root]
        head = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True).strip()
        if head != revision:
            raise ValueError(f"Native source must be pinned to {revision}")
        subprocess.run([*command, "diff", "--quiet", "HEAD", "--"], check=True)
    if not re.fullmatch(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*", config.module):
        raise ValueError("The image must configure its installed camera module")
    for name in ("python", "egohand_python", "droid_python", "droid_moge_checkpoint"):
        if not Path(getattr(config, name)).is_file():
            raise ValueError(f"The HF Jobs image is missing {name}")
    assets = verify_assets(config)
    environment = os.environ.copy()
    for key, value in assets.get("environment", {}).items():
        if key not in {"HF_HOME", "TORCH_HOME", "LD_LIBRARY_PATH"}:
            raise ValueError("Unexpected native release environment option")
        environment[key] = value
    environment.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="0",
        PYTHONPATH=os.pathsep.join(
            [str(Path(__file__).resolve().parents[3]), str(Path(config.root) / "src")]
        ),
    )
    environment.pop("HF_TOKEN", None)
    environment.pop("ANNOTATION_MANIFEST", None)
    if environment.get("DROID_LIBRARY_PATH"):
        environment["LD_LIBRARY_PATH"] = environment["DROID_LIBRARY_PATH"]
    return environment


class RequestProcess:
    """JSON-lines request IDs permit two clips in flight without sharing GPU contexts."""

    def __init__(self, command, *, cwd, environment, log, window=2):
        self.log_path = Path(log)
        self.log = self.log_path.open("w")  # noqa: SIM115 - lifetime is owned by close(), like the subprocess
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.lock = threading.Lock()
        self.pending: dict[str, Future] = {}
        self.window = threading.BoundedSemaphore(window)
        self.closed = False
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            if self.process.stdout is None:
                raise RuntimeError("The native worker must expose its response pipe")
            for line in self.process.stdout:
                result = json.loads(line)
                with self.lock:
                    future = self.pending.pop(result["id"])
                if result.get("status") == "ok":
                    future.set_result(result["result"])
                else:
                    future.set_exception(
                        RuntimeError(result.get("error", "Native annotation failed; inspect its log"))
                    )
        except BaseException as error:
            self._fail_pending(error)
        finally:
            with self.lock:
                self.closed = True
            with self.log_path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - 4000))
                detail = stream.read().decode(errors="replace")
            self._fail_pending(
                RuntimeError("The native worker exited before returning all requests\n" + detail)
            )

    def _fail_pending(self, error):
        with self.lock:
            pending, self.pending = self.pending, {}
        for future in pending.values():
            if not future.done():
                future.set_exception(error)

    def submit(self, payload) -> Future:
        self.window.acquire()
        future: Future = Future()
        future.add_done_callback(lambda _: self.window.release())
        request_id = uuid.uuid4().hex
        try:
            with self.lock:
                if self.closed or self.process.poll() is not None or self.process.stdin is None:
                    raise RuntimeError("The native worker is unavailable; inspect the Job logs")
                self.pending[request_id] = future
                self.process.stdin.write(json.dumps({**payload, "id": request_id}) + "\n")
                self.process.stdin.flush()
        except BaseException as error:
            with self.lock:
                self.pending.pop(request_id, None)
            future.set_exception(error)
        return future

    def close(self):
        with self.lock:
            self.closed = True
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self.reader.join(timeout=5)
        self._fail_pending(RuntimeError("Native worker closed"))
        self.log.close()


class HandAnnotator:
    def __init__(self, config: HandBackendConfig, work: Path):
        self.config = config
        work.mkdir(parents=True, exist_ok=True)
        environment = provisioned_environment(config)
        self.client = RequestProcess(
            [config.python, "-m", "lerobot.annotations.hands.bridge", json.dumps(asdict(config))],
            cwd=config.root,
            environment=environment,
            log=work / "native.log",
        )

    @property
    def process(self):
        return self.client.process

    def annotate(self, video: Path, calibration: dict, directory: Path, *, orientation="sensor") -> dict:
        try:
            return self.client.submit(
                {
                    "video": str(video.resolve()),
                    "calibration": calibration,
                    "directory": str(directory.resolve()),
                    "orientation": orientation,
                }
            ).result(timeout=self.config.timeout_s)
        except TimeoutError:
            self.close()
            raise TimeoutError("Native EgoHand annotation timed out") from None

    def close(self):
        self.client.close()
