import torch

class NoOpRobot:
    """
   Robot that does nothing. 
    """
    cameras = {}
    leader_arms = []
    follower_arms = []
    robot_type: str | None = "no_op"

    @property
    def has_camera(self):
        return False
    
    @property
    def num_cameras(self):
        return 0
    
    @property
    def camera_features(self) -> dict:
        return {}
    
    @property
    def motor_features(self) -> dict:
        return {}
    
    @property
    def is_connected(self):
        return True

    def connect(self):
        pass

    def disconnect(self):
        pass

    def run_calibration(self):
        pass

    def reset(self):
        pass

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return {}, {}

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        return {}

    def send_action(self, action: torch.Tensor):
        """The provided action is expected to be a vector."""
        pass
