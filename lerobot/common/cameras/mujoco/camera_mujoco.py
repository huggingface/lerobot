# sim_camera.py -----------------------------------------------------------
import mujoco
import numpy as np
from lerobot.common.cameras.opencv.camera_opencv import OpenCVCamera  # for the interface only
from .configuration_mujoco import MuJoCoCameraConfig

class MuJoCoCamera(OpenCVCamera):          # subclasses to reuse defaults (fps, etc.)
    def __init__(self, config_or_model, data=None, width: int = 640, height: int = 480,
                 cam: str | int | None = None):
        # Support both config object and direct parameters for backward compatibility
        if isinstance(config_or_model, MuJoCoCameraConfig):
            self.config = config_or_model
            self.m = self.config.model
            self.d = self.config.data
            self.cam = self.config.cam
            self.width = self.config.width or width
            self.height = self.config.height or height
            self.fps = self.config.fps or 30
        else:
            # Backward compatibility: first argument is model
            self.config = None
            self.m = config_or_model
            self.d = data
            self.cam = cam
            self.width = width
            self.height = height
            self.fps = 30                     # LeRobot uses this field
        
        self.r = mujoco.Renderer(self.m, self.width, self.height)
        
        # Set initial robot pose to "home" if keyframe exists
        if self.m.nkey > 0:
            self.d.qpos[:] = self.m.key_qpos[0]  # Use first keyframe ("home")
        
        # Step simulation to initialize properly
        mujoco.mj_forward(self.m, self.d)

    # nothing to physically connect
    def connect(self): ...
    
    def _postprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply color conversion and rotation to the image based on config settings."""
        import cv2
        from lerobot.common.cameras.configs import ColorMode, Cv2Rotation
        
        processed_image = image
        
        # Handle color mode conversion (MuJoCo renders in RGB by default)
        if self.config.color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
        
        # Handle rotation
        if self.config.rotation == Cv2Rotation.ROTATE_90:
            processed_image = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
        elif self.config.rotation == Cv2Rotation.ROTATE_180:
            processed_image = cv2.rotate(processed_image, cv2.ROTATE_180)
        elif self.config.rotation == Cv2Rotation.ROTATE_270:
            processed_image = cv2.rotate(processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return processed_image

    def read(self, color_mode=None):
        """Read a frame synchronously (same as async_read for simulation)."""
        return self.async_read()

    def async_read(self, timeout_ms: int = 0):
        # Just update and render the current state
        self.r.update_scene(self.d, camera=self.cam)
        rgb = self.r.render()             # returns RGB uint8
        image = np.asarray(rgb)
        
        # Apply color mode and rotation if specified
        if hasattr(self, 'config') and isinstance(self.config, MuJoCoCameraConfig):
            image = self._postprocess_image(image)
        
        return image

    def disconnect(self):
        self.r.close()
