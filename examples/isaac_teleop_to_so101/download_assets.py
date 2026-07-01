# !/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Download the SO-101 URDF and meshes used by ``teleoperate.py``.

``teleoperate.py`` loads ``./SO101/so101_new_calib.urdf`` (relative to this
folder) plus the STL meshes it references. This script fetches the URDF from the
upstream `SO-ARM100 <https://github.com/TheRobotStudio/SO-ARM100>`_ repo, parses
out the mesh files it points at, and downloads everything into
``SO101/`` (and ``SO101/assets/``) next to this script.

It uses only the Python standard library, so it has no extra dependencies, and
skips files already present unless ``--force`` is passed.

Usage::

    python download_assets.py            # download into ./SO101
    python download_assets.py --force    # re-download even if present
"""

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Upstream SO-101 simulation assets, tracked from the moving ``main`` branch (NOT
# pinned). There is no release to pin to: the repo's only tag (v0.1.1, May 2024)
# predates the SO-101 and has no Simulation/ dir. The SO101/ layout and the
# so101_new_calib.urdf filename have been stable on main since the assets landed
# (2025-05), so tracking main is fine here; if upstream ever moves the path the
# mesh-reference parse below fails loudly. To freeze the assets, replace ``main``
# with a commit SHA.
RAW_BASE = "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101"
URDF_NAME = "so101_new_calib.urdf"

# Where teleoperate.py expects the assets, relative to this file.
DEST_DIR = Path(__file__).parent / "SO101"


def _download(url: str, dest: Path, force: bool) -> bool:
    """Download ``url`` to ``dest``. Returns True if a download happened."""
    if dest.exists() and not force:
        print(f"  skip   {dest.name} (already present)")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310  # nosec B310 (trusted host)
            data = response.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Failed to download {url}: HTTP {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"Failed to download {url}: {e.reason}") from e
    dest.write_bytes(data)
    print(f"  ok     {dest.name} ({len(data) / 1024:.0f} KiB)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download files even if present")
    args = parser.parse_args()

    print(f"Downloading SO-101 assets into {DEST_DIR}/")

    # 1. The URDF itself.
    urdf_path = DEST_DIR / URDF_NAME
    _download(f"{RAW_BASE}/{URDF_NAME}", urdf_path, args.force)

    # 2. The meshes it references, parsed from the URDF's <mesh filename="..."/>
    #    entries so we only fetch what the example actually loads.
    urdf_text = urdf_path.read_text()
    mesh_rel_paths = sorted(set(re.findall(r'filename="([^"]+)"', urdf_text)))
    if not mesh_rel_paths:
        raise SystemExit(f"No mesh references found in {urdf_path}; upstream layout may have changed.")

    print(f"Found {len(mesh_rel_paths)} mesh reference(s) in {URDF_NAME}:")
    for rel in mesh_rel_paths:
        # rel is like "assets/base_so101_v2.stl", relative to the URDF dir.
        _download(f"{RAW_BASE}/{rel}", DEST_DIR / rel, args.force)

    print(f"\nDone. URDF ready at {urdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
