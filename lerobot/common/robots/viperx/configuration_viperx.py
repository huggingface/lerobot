from dataclasses import dataclass, field

from lerobot.common.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("viperx")
@dataclass
class ViperXRobotConfig(RobotConfig):
    # /!\ FOR SAFETY, READ THIS /!\
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    # For Aloha, for every goal position request, motor rotations are capped at 5 degrees by default.
    # When you feel more confident with teleoperation or running the policy, you can extend
    # this safety limit and even removing it by setting it to `null`.
    # Also, everything is expected to work safely out-of-the-box, but we highly advise to
    # first try to teleoperate the grippers only (by commenting out the rest of the motors in this yaml),
    # then to gradually add more motors (by uncommenting), until you can teleoperate both arms fully
    max_relative_target: int | None = 5

    waist: tuple = (1, "xm540-w270")
    shoulder: tuple = (2, "xm540-w270")
    shoulder_shadow: tuple = (3, "xm540-w270")
    elbow: tuple = (4, "xm540-w270")
    elbow_shadow: tuple = (5, "xm540-w270")
    forearm_roll: tuple = (6, "xm540-w270")
    wrist_angle: tuple = (7, "xm540-w270")
    wrist_rotate: tuple = (8, "xm430-w350")
    gripper: tuple = (9, "xm430-w350")

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # Troubleshooting: If one of your IntelRealSense cameras freeze during
    # data recording due to bandwidth limit, you might need to plug the camera
    # on another USB hub or PCIe card.

    mock: bool = False
