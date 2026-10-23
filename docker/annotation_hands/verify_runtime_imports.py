"""Exercise native startup imports on CPU without models, recordings or inference."""

import importlib
import os
import runpy
import sys
from pathlib import Path


def verify_checkpoint_alias(root: Path):
    """Retain the frozen checkpoint path check from the wrapper's real cwd."""
    frozen = root.parent / "release-195fps-verified/DROID-W/pretrained/droid.pth"
    mounted = root.parents[2] / "workspace/hand-assets/camera/droid.pth"
    # Job assets are intentionally absent at image build time. resolve() checks
    # the symlink chain without loading or distributing any checkpoint bytes.
    if (root / "pretrained/droid.pth").resolve() != frozen.resolve():
        raise RuntimeError("MoGe-on wrapper does not resolve the frozen DROID checkpoint")
    if frozen.resolve() != mounted.resolve():
        raise RuntimeError("Frozen DROID checkpoint does not resolve the Job asset mount")


def main():
    """Import the actual camera bootstrap or concrete hand inference entry points."""
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("Run startup import verification with CUDA_VISIBLE_DEVICES empty")
    if sys.argv[1] == "camera":
        os.environ.update(DROIDW_LEAN_STARTUP="1", MPLBACKEND="Agg")
        root = Path("/opt/hand-camera/DROID-on")
        verify_checkpoint_alias(root)
        sys.path.insert(0, str(root))
        # The frozen bootstrap installs its wrappers at import time. A distinct
        # run name leaves the socket service, model loading and inference idle.
        runpy.run_path(str(root / "grounded_worker.py"), run_name="hand_startup_import_check")
        for module in (
            "run",
            "src.config",
            "src.motion_filter",
            "src.frontend",
            "src.utils.datasets",
            "src.utils.mono_priors.metric_depth_estimators",
            "lerobot_camera_runtime.infra.droidw_moge",
        ):
            importlib.import_module(module)
    elif sys.argv[1] == "egohand":
        for module in ("gem.inference.pose", "gem.common.detectors.hand"):
            importlib.import_module(module)
    else:
        raise ValueError("Choose camera or egohand startup imports")
    print(f"Verified complete {sys.argv[1]} startup imports on CPU; no models loaded", flush=True)


if __name__ == "__main__":
    main()
