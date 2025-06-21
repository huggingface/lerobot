from .config import TeleoperatorConfig
from .teleoperator import Teleoperator
from .utils import make_teleoperator_from_config

"""Teleoperator package initializer.

This sets a safe default for `pynput` when running in SSH/headless sessions so
that any downstream `import pynput` that happens during the recursive imports
below does **not** try to use the X-based backend (which requires the RECORD
extension and crashes with `record_create_context`).

TODO(jackvial): remove this before merging once a proper headless strategy is
implemented upstream.
"""

import os

# Force a safe backend for pynput very early, before any of the sub-modules
# potentially import it.
if ("DISPLAY" not in os.environ or "SSH_CONNECTION" in os.environ) and "PYNPUT_BACKEND" not in os.environ:
    os.environ["PYNPUT_BACKEND"] = "dummy"
