from dataclasses import dataclass, field
from lerobot.common.robots.config import RobotConfig
from lerobot.common.cameras.configs import CameraConfig


@RobotConfig.register_subclass("so101_mujoco")
@dataclass
class SO101SimConfig(RobotConfig):
    """Configuration for the SO100 simulated robot."""
    type: str = "so101_mujoco"
    mjcf_path: str = "lerobot-kinematics/examples/SO101/scene.xml"
    joint_names: list[str] = field(default_factory=lambda: ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"])
    n_substeps: int = 10
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    start_calibrated: bool = True
    show_viewer: bool = True
    enable_rerun: bool = True
    rerun_session_name: str = "so101_mujoco"
    # Joint mapping from dataset names to simulation names with offsets
    joint_mapping: dict[str, tuple[str, float]] = field(default_factory=lambda: {
        "shoulder_pan.pos": ("Rotation", 0.0),
        "shoulder_lift.pos": ("Pitch", -90.0),
        "elbow_flex.pos": ("Elbow", 100.0),
        "wrist_flex.pos": ("Wrist_Pitch", 20.0),
        "wrist_roll.pos": ("Wrist_Roll", -45.0),
        "gripper.pos": ("Jaw", 0.0),
    })
    # Cube randomization settings
    randomize_cube_position: bool = True
    cube_base_position: list[float] = field(default_factory=lambda: [-0.00, -0.32, 0.016])
    cube_randomization_radius: float = 0.05  # 4cm radius