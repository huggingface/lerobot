# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Best-effort software diagnostics, captured once as this process starts.

The revision describes the source checkout at import, not checkpoint contents.
Installed wheels without a source checkout honestly report no revision. These
fields are diagnostic; protocol and execution contracts decide compatibility.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from lerobot import __version__


@dataclass(frozen=True)
class SoftwareBuild:
    """The version and available source revision loaded by this process."""

    lerobot_version: str
    revision: str | None = None
    dirty: bool | None = None


def _capture_build(source_root: Path) -> SoftwareBuild:
    # Only inspect the checkout containing this module. Running from a wheel
    # inside an unrelated Git project must not report that project's revision.
    if not (source_root / ".git").exists():
        return SoftwareBuild(__version__)
    try:
        revision = subprocess.run(  # nosec B603, B607
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=1,
        ).stdout.strip()
        status = subprocess.run(  # nosec B603, B607
            ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True,
            text=True,
            check=True,
            timeout=1,
        ).stdout
        return SoftwareBuild(__version__, revision=revision, dirty=bool(status))
    except (OSError, subprocess.SubprocessError):
        # Missing Git, a source archive or a slow checkout must not prevent use.
        return SoftwareBuild(__version__)


# Do not query Git or installed distribution metadata when answering a later
# DESCRIBE request: an operator may update the checkout while this server runs.
SOFTWARE_BUILD = _capture_build(Path(__file__).resolve().parents[3])
