#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fetch and cache the Metal arm URDF that backs the leader's gravity-compensation model.

The URDF is not vendored into LeRobot: it belongs to the arm's own description repository, and
LeRobot ships no other robot descriptions (`lerobot_find_joint_limits` likewise asks the user to
supply one). It is fetched once, verified against a checksum, and cached under
``HF_LEROBOT_HOME/metal`` so later runs are offline.
"""

import hashlib
import logging
import urllib.error
import urllib.request
from pathlib import Path

from lerobot.utils.constants import HF_LEROBOT_HOME

logger = logging.getLogger(__name__)

# Pinned to a commit, not the `humble` branch tip. The inertials in this file set the feedforward
# torques the leader streams into a human's hand, so an edit upstream must never silently retune
# the arm; bumping the arm's dynamics has to be a visible change to this constant.
METAL_URDF_COMMIT = "ef4181f1305cbcfc63431d3bcfb96f5fb7f72763"

# NOTE: `metal_sdk/example/urdf/`, NOT `metal_ros2/src/metal_description/urdf/`. The description
# package models the gripper's two prismatic jaws as joints, giving nq=8; this variant stops at
# the 6 arm revolutes (nq=6) with the gripper's mass lumped into Link6, which is the model
# MetalGravityModel's `q[:6]` expects. Passing the description URDF raises inside Pinocchio's
# computeGeneralizedGravity on every tick.
METAL_URDF_URL = (
    "https://raw.githubusercontent.com/makermods-robotics/metal-python-ros/"
    f"{METAL_URDF_COMMIT}/metal_sdk/example/urdf/metal_with_gripper.urdf"
)
METAL_URDF_SHA256 = "faac0ba624b28cf531834be0bf4eb90595ae78648dbfe12058b8ec656b65f7ef"

METAL_URDF_FILENAME = "metal_with_gripper.urdf"
METAL_URDF_CACHE_DIR = HF_LEROBOT_HOME / "metal"

DOWNLOAD_TIMEOUT_SEC = 30.0


class MetalUrdfError(RuntimeError):
    """The Metal URDF could not be fetched or failed its integrity check."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _download() -> bytes:
    try:
        # METAL_URDF_URL is a module-level https:// constant, never caller-supplied, and the
        # payload is sha256-checked by the caller -- so the scheme audit B310/S310 warns about
        # does not apply here.
        with urllib.request.urlopen(  # noqa: S310 # nosec B310
            METAL_URDF_URL, timeout=DOWNLOAD_TIMEOUT_SEC
        ) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise MetalUrdfError(
            f"Could not download the Metal URDF from {METAL_URDF_URL}: {exc}. "
            "Gravity compensation cannot start without it. If this machine has no network "
            "access, pass a local copy with `--teleop.urdf_path=/path/to/metal_with_gripper.urdf`."
        ) from exc


def metal_urdf_path(cache_dir: Path | None = None) -> Path:
    """Return a local path to the Metal URDF, downloading it on first use.

    A cached file whose checksum no longer matches is treated as corrupt and re-downloaded once.
    Raises `MetalUrdfError` if the file cannot be obtained or fails verification, so the caller
    can refuse to energize the arm rather than run it without gravity compensation.
    """
    cache_dir = METAL_URDF_CACHE_DIR if cache_dir is None else Path(cache_dir)
    cached = cache_dir / METAL_URDF_FILENAME

    if cached.is_file():
        if _sha256(cached.read_bytes()) == METAL_URDF_SHA256:
            return cached
        logger.warning(f"Cached Metal URDF at {cached} failed its checksum; re-downloading.")

    logger.info(f"Downloading the Metal URDF from {METAL_URDF_URL}")
    payload = _download()

    digest = _sha256(payload)
    if digest != METAL_URDF_SHA256:
        raise MetalUrdfError(
            f"Metal URDF checksum mismatch: expected {METAL_URDF_SHA256}, got {digest}. "
            "Refusing to build a gravity model from an unexpected description."
        )

    # Write via a temporary file so a crash mid-write cannot leave a truncated URDF in the cache
    # that would then be re-verified (and rejected) on every subsequent run.
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(".urdf.tmp")
    tmp.write_bytes(payload)
    tmp.replace(cached)
    logger.info(f"Cached the Metal URDF at {cached}")
    return cached
