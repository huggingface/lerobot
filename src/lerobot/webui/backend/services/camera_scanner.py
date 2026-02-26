"""Camera scanning service wrapping lerobot_find_cameras logic."""

import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from lerobot.webui.backend.models.setup import CameraInfo, CameraPreview


class CameraScannerService:
    """Service for scanning and detecting cameras."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize CameraScannerService.

        Args:
            output_dir: Directory for storing preview images. Defaults to outputs/camera_previews.
        """
        if output_dir is None:
            repo_root = Path(__file__).parent.parent.parent.parent.parent.parent
            output_dir = repo_root / "outputs" / "camera_previews"

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_external_camera(camera: dict) -> bool:
        """Determine if a camera from system_profiler is an external USB camera.

        Built-in/non-external cameras include:
        - MacBook built-in cameras (FaceTime, MacBook Pro Camera, etc.)
        - iPhone Continuity cameras (model ID contains "iPhone")
        - iPad Continuity cameras (model ID contains "iPad")

        Args:
            camera: A camera dict from system_profiler SPCameraDataType JSON output.

        Returns:
            True if the camera is an external USB camera, False otherwise.
        """
        name = camera.get("_name", "").lower()
        model_id = camera.get("spcamera_model-id", "").lower()

        # Explicit built-in flag (some macOS versions)
        if camera.get("_is_builtin", "") == "yes":
            return False

        # MacBook built-in cameras
        if "macbook" in name or "facetime" in name or "built-in" in name:
            return False
        if "macbook" in model_id or "facetime" in model_id:
            return False

        # iPhone/iPad Continuity Camera
        if "iphone" in model_id or "ipad" in model_id:
            return False

        return True

    def _get_system_camera_info(self) -> List[dict]:
        """Get ordered camera list from macOS system_profiler.

        The order matches OpenCV camera indices.

        Returns:
            List of camera dicts from system_profiler, or empty list on non-macOS / errors.
        """
        if platform.system() != "Darwin":
            return []

        try:
            result = subprocess.run(
                ["system_profiler", "SPCameraDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            data = json.loads(result.stdout)

            for camera_list in data.values():
                if isinstance(camera_list, list):
                    return camera_list

            return []
        except Exception:
            return []

    def list_cameras(self, max_cameras: int = 10, exclude_builtin: bool = False) -> List[CameraInfo]:
        """List all available OpenCV cameras.

        Args:
            max_cameras: Maximum number of camera indices to check.
            exclude_builtin: If True, only return external USB cameras (macOS only).

        Returns:
            List of CameraInfo objects for detected cameras.
        """
        system_cameras = self._get_system_camera_info()
        cameras = []

        for index in range(max_cameras):
            cap = cv2.VideoCapture(index)

            if cap.isOpened():
                # Use system_profiler name if available (same ordering as OpenCV indices)
                is_external = True
                if index < len(system_cameras):
                    cam_info = system_cameras[index]
                    camera_name = cam_info.get("_name", f"Camera {index}")
                    is_external = self._is_external_camera(cam_info)
                else:
                    camera_name = f"Camera {index}"

                cap.release()

                if exclude_builtin and not is_external:
                    continue

                cameras.append(
                    CameraInfo(
                        index=index,
                        name=camera_name,
                        backend="opencv",
                        is_builtin=not is_external,
                    )
                )

        return cameras

    def capture_preview(
        self, camera_indices: Optional[List[int]] = None, record_time_s: float = 2.0
    ) -> Dict[int, CameraPreview]:
        """Capture preview images from cameras.

        Args:
            camera_indices: List of camera indices to capture. If None, captures all detected cameras.
            record_time_s: Time to wait before capturing image (for camera to warm up).

        Returns:
            Dictionary mapping camera index to CameraPreview.
        """
        import time

        # Clear old previews
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Detect cameras if not specified
        if camera_indices is None:
            detected = self.list_cameras()
            camera_indices = [cam.index for cam in detected]

        previews = {}

        for index in camera_indices:
            cap = cv2.VideoCapture(index)

            if not cap.isOpened():
                continue

            # Wait for camera to warm up
            time.sleep(record_time_s)

            # Capture frame
            ret, frame = cap.read()

            if ret:
                image_path = self.output_dir / f"camera_{index}.jpg"
                cv2.imwrite(str(image_path), frame)

                previews[index] = CameraPreview(
                    index=index,
                    image_path=str(image_path),
                    image_url=f"/api/setup/cameras/preview/{index}",  # Relative URL
                )

            cap.release()

        return previews
