import cv2
import numpy as np


JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def build_residual_observation(obs):
    state = np.array(
        [obs[k] for k in JOINT_KEYS],
        dtype=np.float32,
    )

    front = np.asarray(obs["front"])
    wrist = np.asarray(obs["wrist"])

    front = cv2.resize(front, (32, 32))
    wrist = cv2.resize(wrist, (32, 32))

    if front.ndim == 3:
        front = cv2.cvtColor(front, cv2.COLOR_RGB2GRAY)

    if wrist.ndim == 3:
        wrist = cv2.cvtColor(wrist, cv2.COLOR_RGB2GRAY)

    front = front.astype(np.float32) / 255.0
    wrist = wrist.astype(np.float32) / 255.0

    observation = np.concatenate(
        [
            state,
            front.reshape(-1),
            wrist.reshape(-1),
        ]
    ).astype(np.float32)

    return observation
