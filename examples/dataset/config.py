XVLA_SOFT_FOLD_FEATURES = {
    "observation.images.cam_high": {
        "dtype": "video",
        "names": ["height", "width", "channels"],
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.cam_left_wrist": {
        "dtype": "video",
        "names": ["height", "width", "channels"],
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.cam_right_wrist": {
        "dtype": "video",
        "names": ["height", "width", "channels"],
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },

    "observation.states.eef_euler": {
        "dtype": "float32",
        "shape": (14,),   # 14 = 7 joints per arm × 2 arms OR 14-d state representation
        "names": {"values": [f"eef_euler_{i}" for i in range(14)]},
    },

    "observation.states.eef_quaternion": {
        "dtype": "float32",
        "shape": (16,),   # 16 = 8 quaternion floats per arm × 2 arms
        "names": {"values": [f"eef_quat_{i}" for i in range(16)]},
    },

    "observation.states.eef_6d": {
        "dtype": "float32",
        "shape": (20,),   # 20 = pos(3) + rot6d(6) + extra dims
        "names": {"values": [f"eef6d_{i}" for i in range(20)]},
    },

    "observation.states.eef_left_time": {
        "dtype": "float32",
        "shape": (1,),
        "names": {"values": ["eef_left_time"]},
    },

    "observation.states.eef_right_time": {
        "dtype": "float32",
        "shape": (1,),
        "names": {"values": ["eef_right_time"]},
    },

    "observation.states.qpos": {
        "dtype": "float32",
        "shape": (14,),   # 7 per arm × 2 arms
        "names": {"motors": [f"qpos_{i}" for i in range(14)]},
    },

    "observation.states.qvel": {
        "dtype": "float32",
        "shape": (14,),
        "names": {"motors": [f"qvel_{i}" for i in range(14)]},
    },

    "observation.states.effort": {
        "dtype": "float32",
        "shape": (14,),
        "names": {"motors": [f"effort_{i}" for i in range(14)]},
    },

    "observation.states.qpos_left_time": {
        "dtype": "float32",
        "shape": (1,),
        "names": {"values": ["qpos_left_time"]},
    },

    "observation.states.qpos_right_time": {
        "dtype": "float32",
        "shape": (1,),
        "names": {"values": ["qpos_right_time"]},
    },

    "action": {
        "dtype": "float32",
        "shape": (14,),
        "names": {"motors": [f"joint_action_{i}" for i in range(14)]},
    },

    "time_stamp": {
        "dtype": "float32",
        "shape": (1,),
        "names": {"values": ["global_timestamp"]},
    },

}
