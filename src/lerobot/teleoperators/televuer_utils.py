import logging
import numpy as np
import asyncio
import cv2
import os
from pathlib import Path
from multiprocessing import Value, Array, Process, shared_memory
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from vuer import Vuer
from vuer.schemas import ImageBackground, Hands, MotionControllers, WebRTCVideoPlane


@dataclass
class TeleStateData:
    # hand tracking
    left_pinch_state: bool = False         # True if index and thumb are pinching
    left_squeeze_state: bool = False       # True if hand is making a fist
    left_squeeze_value: float = 0.0        # (0.0 ~ 1.0) degree of hand squeeze
    right_pinch_state: bool = False        # True if index and thumb are pinching
    right_squeeze_state: bool = False      # True if hand is making a fist
    right_squeeze_value: float = 0.0       # (0.0 ~ 1.0) degree of hand squeeze

    # controller tracking
    left_trigger_state: bool = False       # True if trigger is actively pressed
    left_squeeze_ctrl_state: bool = False  # True if grip button is pressed
    left_squeeze_ctrl_value: float = 0.0   # (0.0 ~ 1.0) grip pull depth
    left_thumbstick_state: bool = False    # True if thumbstick button is pressed
    left_thumbstick_value: np.ndarray = field(default_factory=lambda: np.zeros(2)) # 2D vector (x, y), normalized
    left_aButton: bool = False             # True if A button is pressed
    left_bButton: bool = False             # True if B button is pressed
    right_trigger_state: bool = False      # True if trigger is actively pressed
    right_squeeze_ctrl_state: bool = False # True if grip button is pressed
    right_squeeze_ctrl_value: float = 0.0  # (0.0 ~ 1.0) grip pull depth
    right_thumbstick_state: bool = False   # True if thumbstick button is pressed
    right_thumbstick_value: np.ndarray = field(default_factory=lambda: np.zeros(2)) # 2D vector (x, y), normalized
    right_aButton: bool = False            # True if A button is pressed
    right_bButton: bool = False            # True if B button is pressed

@dataclass
class TeleData:
    head_pose: np.ndarray       # (4,4) SE(3) pose of head matrix
    left_arm_pose: np.ndarray   # (4,4) SE(3) pose of left arm
    right_arm_pose: np.ndarray  # (4,4) SE(3) pose of right arm
    left_hand_pos: Optional[np.ndarray] = None  # (25,3) 3D positions of left hand joints
    right_hand_pos: Optional[np.ndarray] = None # (25,3) 3D positions of right hand joints
    left_hand_rot: Optional[np.ndarray]  = None # (25,3,3) rotation matrices of left hand joints
    right_hand_rot: Optional[np.ndarray] = None # (25,3,3) rotation matrices of right hand joints
    left_pinch_value: Optional[float] = None    # float (1x.0 ~ 0.0) pinch distance between index and thumb
    right_pinch_value: Optional[float] = None   # float (1x.0 ~ 0.0) pinch distance between index and thumb
    left_trigger_value: Optional[float] = None  # float (10.0 ~ 0.0) trigger pull depth
    right_trigger_value: Optional[float] = None # float (10.0 ~ 0.0) trigger pull depth
    tele_state: TeleStateData = field(default_factory=TeleStateData)


class TeleVuer:
    def __init__(self, binocular: bool, use_hand_tracking: bool, img_shape,
                 img_shm_name=None,
                 left_img_shm_name=None,
                 right_img_shm_name=None,
                 cert_file=None, key_file=None, ngrok=False, webrtc=False,
                 webrtc_offer_url=None):
        """
        TeleVuer class for OpenXR-based XR teleoperate applications.
        This class handles the communication with the Vuer server and manages the shared memory for image and pose data.
        
        Args:
            webrtc_offer_url: URL for WebRTC offer endpoint (e.g., "https://host:8080/offer").
                              Required if webrtc=True. Can also be set via WEBRTC_OFFER_URL env var.
        """
        self.binocular = binocular
        self.use_hand_tracking = use_hand_tracking
        self.webrtc_offer_url = webrtc_offer_url or os.environ.get("WEBRTC_OFFER_URL")
        self.img_height = img_shape[0]
        # Determine per-eye width depending on buffer layout
        self._using_split_shm = bool(left_img_shm_name and right_img_shm_name)
        if self._using_split_shm:
            self.img_width = img_shape[1]
        else:
            if binocular:
                self.img_width = img_shape[1] // 2
            else:
                self.img_width = img_shape[1]
        
        # Use default certs if not provided, relative to this file or expected location
        # For lerobot integration, we might need to rely on the user providing them or looking in a standard path
        if cert_file is None:
             # Fallback to looking in current directory or updated logic needed
             pass 

        if ngrok:
            self.vuer = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)
        else:
            # Note: cert_file and key_file must be valid paths if not using ngrok and SSL is required
            # If they are None, Vuer might default to http or generated certs depending on implementation
            kwargs = dict(queries=dict(grid=False), queue_len=3)
            if cert_file and key_file:
                 kwargs['cert'] = cert_file
                 kwargs['key'] = key_file
            self.vuer = Vuer(host='0.0.0.0', **kwargs)

        self.vuer.add_handler("CAMERA_MOVE")(self.on_cam_move)
        if self.use_hand_tracking:
            self.vuer.add_handler("HAND_MOVE")(self.on_hand_move)
        else:
            self.vuer.add_handler("CONTROLLER_MOVE")(self.on_controller_move)

        # Image shared memory setup
        if self._using_split_shm:
            left_shm = shared_memory.SharedMemory(name=left_img_shm_name)
            right_shm = shared_memory.SharedMemory(name=right_img_shm_name)
            self.left_img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=left_shm.buf)
            self.right_img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=right_shm.buf)
            self._left_shm = left_shm
            self._right_shm = right_shm
        else:
            if img_shm_name is None:
                raise ValueError("img_shm_name must be provided when left/right shared memories are not used.")
            existing_shm = shared_memory.SharedMemory(name=img_shm_name)
            self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=existing_shm.buf)
            self._img_shm = existing_shm

        self.webrtc = webrtc
        if self.binocular and not self.webrtc:
            self.vuer.spawn(start=False)(self.main_image_binocular)
        elif not self.binocular and not self.webrtc:
            self.vuer.spawn(start=False)(self.main_image_monocular)
        elif self.webrtc:
            self.vuer.spawn(start=False)(self.main_image_webrtc)

        self.head_pose_shared = Array('d', 16, lock=True)
        self.left_arm_pose_shared = Array('d', 16, lock=True)
        self.right_arm_pose_shared = Array('d', 16, lock=True)
        if self.use_hand_tracking:
            self.left_hand_position_shared = Array('d', 75, lock=True)
            self.right_hand_position_shared = Array('d', 75, lock=True)
            self.left_hand_orientation_shared = Array('d', 25 * 9, lock=True)
            self.right_hand_orientation_shared = Array('d', 25 * 9, lock=True)

            self.left_pinch_state_shared = Value('b', False, lock=True)
            self.left_pinch_value_shared = Value('d', 0.0, lock=True)
            self.left_squeeze_state_shared = Value('b', False, lock=True)
            self.left_squeeze_value_shared = Value('d', 0.0, lock=True)

            self.right_pinch_state_shared = Value('b', False, lock=True)
            self.right_pinch_value_shared = Value('d', 0.0, lock=True)
            self.right_squeeze_state_shared = Value('b', False, lock=True)
            self.right_squeeze_value_shared = Value('d', 0.0, lock=True)
        else:
            self.left_trigger_state_shared = Value('b', False, lock=True)
            self.left_trigger_value_shared = Value('d', 0.0, lock=True)
            self.left_squeeze_state_shared = Value('b', False, lock=True)
            self.left_squeeze_value_shared = Value('d', 0.0, lock=True)
            self.left_thumbstick_state_shared = Value('b', False, lock=True)
            self.left_thumbstick_value_shared = Array('d', 2, lock=True)
            self.left_aButton_shared = Value('b', False, lock=True)
            self.left_bButton_shared = Value('b', False, lock=True)

            self.right_trigger_state_shared = Value('b', False, lock=True)
            self.right_trigger_value_shared = Value('d', 0.0, lock=True)
            self.right_squeeze_state_shared = Value('b', False, lock=True)
            self.right_squeeze_value_shared = Value('d', 0.0, lock=True)
            self.right_thumbstick_state_shared = Value('b', False, lock=True)
            self.right_thumbstick_value_shared = Array('d', 2, lock=True)
            self.right_aButton_shared = Value('b', False, lock=True)
            self.right_bButton_shared = Value('b', False, lock=True)

        self.process = Process(target=self.vuer_run)
        self.process.daemon = True
        self.process.start()

    def vuer_run(self):
        self.vuer.run()

    async def on_cam_move(self, event, session, fps=60):
        try:
            with self.head_pose_shared.get_lock():
                self.head_pose_shared[:] = event.value["camera"]["matrix"]
        except Exception as e:
            logger.debug(f"Error handling camera move event: {e}")

    async def on_controller_move(self, event, session, fps=60):
        try:
            with self.left_arm_pose_shared.get_lock():
                self.left_arm_pose_shared[:] = event.value["left"]
            with self.right_arm_pose_shared.get_lock():
                self.right_arm_pose_shared[:] = event.value["right"]

            left_controller_state = event.value["leftState"]
            right_controller_state = event.value["rightState"]

            def extract_controller_states(state_dict, prefix):
                # trigger
                with getattr(self, f"{prefix}_trigger_state_shared").get_lock():
                    getattr(self, f"{prefix}_trigger_state_shared").value = bool(state_dict.get("trigger", False))
                with getattr(self, f"{prefix}_trigger_value_shared").get_lock():
                    getattr(self, f"{prefix}_trigger_value_shared").value = float(state_dict.get("triggerValue", 0.0))
                # squeeze
                with getattr(self, f"{prefix}_squeeze_state_shared").get_lock():
                    getattr(self, f"{prefix}_squeeze_state_shared").value = bool(state_dict.get("squeeze", False))
                with getattr(self, f"{prefix}_squeeze_value_shared").get_lock():
                    getattr(self, f"{prefix}_squeeze_value_shared").value = float(state_dict.get("squeezeValue", 0.0))
                # thumbstick
                with getattr(self, f"{prefix}_thumbstick_state_shared").get_lock():
                    getattr(self, f"{prefix}_thumbstick_state_shared").value = bool(state_dict.get("thumbstick", False))
                with getattr(self, f"{prefix}_thumbstick_value_shared").get_lock():
                    getattr(self, f"{prefix}_thumbstick_value_shared")[:] = state_dict.get("thumbstickValue", [0.0, 0.0])
                # buttons
                with getattr(self, f"{prefix}_aButton_shared").get_lock():
                    getattr(self, f"{prefix}_aButton_shared").value = bool(state_dict.get("aButton", False))
                with getattr(self, f"{prefix}_bButton_shared").get_lock():
                    getattr(self, f"{prefix}_bButton_shared").value = bool(state_dict.get("bButton", False))

            extract_controller_states(left_controller_state, "left")
            extract_controller_states(right_controller_state, "right")
        except Exception as e:
            logger.debug(f"Error handling controller move event: {e}")

    async def on_hand_move(self, event, session, fps=60):
        try:
            left_hand_data = event.value["left"]
            right_hand_data = event.value["right"]
            left_hand_state = event.value["leftState"]
            right_hand_state = event.value["rightState"]

            def extract_hand_poses(hand_data, arm_pose_shared, hand_position_shared, hand_orientation_shared):
                with arm_pose_shared.get_lock():
                    arm_pose_shared[:] = hand_data[0:16]

                with hand_position_shared.get_lock():
                    for i in range(25):
                        base = i * 16
                        hand_position_shared[i * 3: i * 3 + 3] = [hand_data[base + 12], hand_data[base + 13], hand_data[base + 14]]

                with hand_orientation_shared.get_lock():
                    for i in range(25):
                        base = i * 16
                        hand_orientation_shared[i * 9: i * 9 + 9] = [
                            hand_data[base + 0], hand_data[base + 1], hand_data[base + 2],
                            hand_data[base + 4], hand_data[base + 5], hand_data[base + 6],
                            hand_data[base + 8], hand_data[base + 9], hand_data[base + 10],
                        ]

            def extract_hand_states(state_dict, prefix):
                # pinch
                with getattr(self, f"{prefix}_pinch_state_shared").get_lock():
                    getattr(self, f"{prefix}_pinch_state_shared").value = bool(state_dict.get("pinch", False))
                with getattr(self, f"{prefix}_pinch_value_shared").get_lock():
                    getattr(self, f"{prefix}_pinch_value_shared").value = float(state_dict.get("pinchValue", 0.0))
                # squeeze
                with getattr(self, f"{prefix}_squeeze_state_shared").get_lock():
                    getattr(self, f"{prefix}_squeeze_state_shared").value = bool(state_dict.get("squeeze", False))
                with getattr(self, f"{prefix}_squeeze_value_shared").get_lock():
                    getattr(self, f"{prefix}_squeeze_value_shared").value = float(state_dict.get("squeezeValue", 0.0))

            extract_hand_poses(left_hand_data, self.left_arm_pose_shared, self.left_hand_position_shared, self.left_hand_orientation_shared)
            extract_hand_poses(right_hand_data, self.right_arm_pose_shared, self.right_hand_position_shared, self.right_hand_orientation_shared)
            extract_hand_states(left_hand_state, "left")
            extract_hand_states(right_hand_state, "right")

        except Exception as e:
            logger.debug(f"Error handling hand move event: {e}")
    
    async def main_image_binocular(self, session, fps=60):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            if self._using_split_shm:
                left_rgb = cv2.cvtColor(self.left_img_array, cv2.COLOR_BGR2RGB)
                right_rgb = cv2.cvtColor(self.right_img_array, cv2.COLOR_BGR2RGB)
                left_frame = left_rgb
                right_frame = right_rgb
            else:
                display_image = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2RGB)
                left_frame = display_image[:, :self.img_width]
                right_frame = display_image[:, self.img_width:]

            session.upsert(
                [
                    ImageBackground(
                        left_frame,
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        layers=1,
                        format="jpeg",
                        quality=100,
                        key="background-left",
                        interpolate=True,
                    ),
                    ImageBackground(
                        right_frame,
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        layers=2,
                        format="jpeg",
                        quality=100,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            await asyncio.sleep(0.016 * 2)

    async def main_image_monocular(self, session, fps=60):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            if self._using_split_shm:
                mono_rgb = cv2.cvtColor(self.left_img_array, cv2.COLOR_BGR2RGB)
            else:
                mono_rgb = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2RGB)
            session.upsert(
                [
                    ImageBackground(
                        mono_rgb,
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        format="jpeg",
                        quality=50,
                        key="background-mono",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            await asyncio.sleep(0.016)

    async def main_image_webrtc(self, session, fps=60):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    showLeft=False,
                    showRight=False
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    showLeft=False,
                    showRight=False,
                )
            )
    
        session.upsert(
            WebRTCVideoPlane(
                src=self.webrtc_offer_url,
                iceServer={},
                key="webrtc",
                aspect=1.778,
                height=7,
            ),
            to="bgChildren",
        )
        while True:
            await asyncio.sleep(1)

    # ==================== common data ====================
    @property
    def head_pose(self):
        with self.head_pose_shared.get_lock():
            return np.array(self.head_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def left_arm_pose(self):
        with self.left_arm_pose_shared.get_lock():
            return np.array(self.left_arm_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def right_arm_pose(self):
        with self.right_arm_pose_shared.get_lock():
            return np.array(self.right_arm_pose_shared[:]).reshape(4, 4, order="F")

    # ==================== Hand Tracking Data ====================
    @property
    def left_hand_positions(self):
        with self.left_hand_position_shared.get_lock():
            return np.array(self.left_hand_position_shared[:]).reshape(25, 3)

    @property
    def right_hand_positions(self):
        with self.right_hand_position_shared.get_lock():
            return np.array(self.right_hand_position_shared[:]).reshape(25, 3)

    @property
    def left_hand_orientations(self):
        with self.left_hand_orientation_shared.get_lock():
            return np.array(self.left_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order="F")

    @property
    def right_hand_orientations(self):
        with self.right_hand_orientation_shared.get_lock():
            return np.array(self.right_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order="F")

    @property
    def left_hand_pinch_state(self):
        with self.left_pinch_state_shared.get_lock():
            return self.left_pinch_state_shared.value

    @property
    def left_hand_pinch_value(self):
        with self.left_pinch_value_shared.get_lock():
            return self.left_pinch_value_shared.value

    @property
    def left_hand_squeeze_state(self):
        with self.left_squeeze_state_shared.get_lock():
            return self.left_squeeze_state_shared.value

    @property
    def left_hand_squeeze_value(self):
        with self.left_squeeze_value_shared.get_lock():
            return self.left_squeeze_value_shared.value

    @property
    def right_hand_pinch_state(self):
        with self.right_pinch_state_shared.get_lock():
            return self.right_pinch_state_shared.value

    @property
    def right_hand_pinch_value(self):
        with self.right_pinch_value_shared.get_lock():
            return self.right_pinch_value_shared.value

    @property
    def right_hand_squeeze_state(self):
        with self.right_squeeze_state_shared.get_lock():
            return self.right_squeeze_state_shared.value

    @property
    def right_hand_squeeze_value(self):
        with self.right_squeeze_value_shared.get_lock():
            return self.right_squeeze_value_shared.value

    # ==================== Controller Data ====================
    @property
    def left_controller_trigger_state(self):
        with self.left_trigger_state_shared.get_lock():
            return self.left_trigger_state_shared.value

    @property
    def left_controller_trigger_value(self):
        with self.left_trigger_value_shared.get_lock():
            return self.left_trigger_value_shared.value

    @property
    def left_controller_squeeze_state(self):
        with self.left_squeeze_state_shared.get_lock():
            return self.left_squeeze_state_shared.value

    @property
    def left_controller_squeeze_value(self):
        with self.left_squeeze_value_shared.get_lock():
            return self.left_squeeze_value_shared.value

    @property
    def left_controller_thumbstick_state(self):
        with self.left_thumbstick_state_shared.get_lock():
            return self.left_thumbstick_state_shared.value

    @property
    def left_controller_thumbstick_value(self):
        with self.left_thumbstick_value_shared.get_lock():
            return np.array(self.left_thumbstick_value_shared[:])

    @property
    def left_controller_aButton(self):
        with self.left_aButton_shared.get_lock():
            return self.left_aButton_shared.value

    @property
    def left_controller_bButton(self):
        with self.left_bButton_shared.get_lock():
            return self.left_bButton_shared.value

    @property
    def right_controller_trigger_state(self):
        with self.right_trigger_state_shared.get_lock():
            return self.right_trigger_state_shared.value

    @property
    def right_controller_trigger_value(self):
        with self.right_trigger_value_shared.get_lock():
            return self.right_trigger_value_shared.value

    @property
    def right_controller_squeeze_state(self):
        with self.right_squeeze_state_shared.get_lock():
            return self.right_squeeze_state_shared.value

    @property
    def right_controller_squeeze_value(self):
        with self.right_squeeze_value_shared.get_lock():
            return self.right_squeeze_value_shared.value

    @property
    def right_controller_thumbstick_state(self):
        with self.right_thumbstick_state_shared.get_lock():
            return self.right_thumbstick_state_shared.value

    @property
    def right_controller_thumbstick_value(self):
        with self.right_thumbstick_value_shared.get_lock():
            return np.array(self.right_thumbstick_value_shared[:])

    @property
    def right_controller_aButton(self):
        with self.right_aButton_shared.get_lock():
            return self.right_aButton_shared.value

    @property
    def right_controller_bButton(self):
        with self.right_bButton_shared.get_lock():
            return self.right_bButton_shared.value


# Constants and helper functions from tv_wrapper.py
def safe_mat_update(prev_mat, mat):
    # Return previous matrix and False flag if the new matrix is non-singular (determinant ≠ 0).
    det = np.linalg.det(mat)
    if not np.isfinite(det) or np.isclose(det, 0.0, atol=1e-6):
        return prev_mat, False
    return mat, True

def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret

def safe_rot_update(prev_rot_array, rot_array):
    dets = np.linalg.det(rot_array)
    if not np.all(np.isfinite(dets)) or np.any(np.isclose(dets, 0.0, atol=1e-6)):
        return prev_rot_array, False
    return rot_array, True

# constants variable
T_TO_UNITREE_HUMANOID_LEFT_ARM = np.array([[1, 0, 0, 0],
                                           [0, 0,-1, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 0, 1]])

T_TO_UNITREE_HUMANOID_RIGHT_ARM = np.array([[1, 0, 0, 0],
                                            [0, 0, 1, 0],
                                            [0,-1, 0, 0],
                                            [0, 0, 0, 1]])

T_TO_UNITREE_HAND = np.array([[0,  0, 1, 0],
                              [-1, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0,  0, 0, 1]])

T_ROBOT_OPENXR = np.array([[ 0, 0,-1, 0],
                           [-1, 0, 0, 0],
                           [ 0, 1, 0, 0],
                           [ 0, 0, 0, 1]])

T_OPENXR_ROBOT = np.array([[ 0,-1, 0, 0],
                           [ 0, 0, 1, 0],
                           [-1, 0, 0, 0],
                           [ 0, 0, 0, 1]])

R_ROBOT_OPENXR = np.array([[ 0, 0,-1],
                           [-1, 0, 0],
                           [ 0, 1, 0]])

R_OPENXR_ROBOT = np.array([[ 0,-1, 0],
                           [ 0, 0, 1],
                           [-1, 0, 0]])

CONST_HEAD_POSE = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 1.5],
                            [0, 0, 1, -0.2],
                            [0, 0, 0, 1]])

CONST_RIGHT_ARM_POSE = np.array([[1, 0, 0, 0.15],
                                 [0, 1, 0, 1.13],
                                 [0, 0, 1, -0.3],
                                 [0, 0, 0, 1]])

CONST_LEFT_ARM_POSE = np.array([[1, 0, 0, -0.15],
                                [0, 1, 0, 1.13],
                                [0, 0, 1, -0.3],
                                [0, 0, 0, 1]])

CONST_HAND_ROT = np.tile(np.eye(3)[None, :, :], (25, 1, 1))

