from lerobot.common.constants import HF_LEROBOT_HOME

LEROBOT_TEST_DIR = HF_LEROBOT_HOME / "_testing"
DUMMY_REPO_ID = "dummy/repo"
DUMMY_ROBOT_TYPE = "dummy_robot"
DUMMY_MOTOR_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
    },
    "state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
    },
}
DUMMY_CAMERA_FEATURES = {
    "laptop": {"shape": (480, 640, 3), "names": ["height", "width", "channels"], "info": None},
    "phone": {"shape": (480, 640, 3), "names": ["height", "width", "channels"], "info": None},
}
DEFAULT_FPS = 30
DUMMY_VIDEO_INFO = {
    "video.fps": DEFAULT_FPS,
    "video.codec": "av1",
    "video.pix_fmt": "yuv420p",
    "video.is_depth_map": False,
    "has_audio": False,
}
DUMMY_CHW = (3, 96, 128)
DUMMY_HWC = (96, 128, 3)
