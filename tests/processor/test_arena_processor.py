# tests/processor/test_arena_processor.py
import pytest
import torch

from lerobot.envs.configs import IsaaclabArenaEnv
from lerobot.processor.env_processor import IsaaclabArenaProcessorStep
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, OBS_STR

# =============================================================================
# Test Configuration - derived from IsaaclabArenaEnv defaults
# =============================================================================
# Create a reference config to get default values
_REF_CONFIG = IsaaclabArenaEnv()

# Dimensions from config
STATE_DIM = _REF_CONFIG.state_dim  # 54
ACTION_DIM = _REF_CONFIG.action_dim  # 36
CAMERA_HEIGHT = _REF_CONFIG.camera_height  # 512
CAMERA_WIDTH = _REF_CONFIG.camera_width  # 512

# Test-specific constants
BATCH_SIZE = 2


class TestIsaaclabArenaProcessorStep:
    """Tests for IsaaclabArenaProcessorStep."""

    @pytest.fixture
    def processor(self):
        return IsaaclabArenaProcessorStep()

    @pytest.fixture
    def sample_observation(self):
        """Create a sample IsaacLab Arena observation."""
        return {
            f"{OBS_STR}.policy": {
                "robot_joint_pos": torch.randn(BATCH_SIZE, STATE_DIM),
                "actions": torch.randn(BATCH_SIZE, ACTION_DIM),
            },
            f"{OBS_STR}.camera_obs": {
                "robot_pov_cam_rgb": torch.randint(
                    0,
                    255,
                    (BATCH_SIZE, CAMERA_HEIGHT, CAMERA_WIDTH, 3),
                    dtype=torch.uint8,
                ),
            },
        }

    def test_processor_processes_robot_state(self, processor, sample_observation):
        """Test that robot state is correctly extracted."""
        processed = processor.observation(sample_observation)

        assert OBS_STATE in processed
        assert processed[OBS_STATE].shape == (BATCH_SIZE, STATE_DIM)
        assert processed[OBS_STATE].dtype == torch.float32

    def test_processor_processes_camera_observation(self, processor, sample_observation):
        """Test that camera observations are correctly processed."""
        processed = processor.observation(sample_observation)

        # Check camera key exists (with _rgb suffix stripped)
        assert f"{OBS_IMAGES}.robot_pov_cam" in processed

        img = processed[f"{OBS_IMAGES}.robot_pov_cam"]
        # Shape: (B, C, H, W)
        assert img.shape == (BATCH_SIZE, 3, CAMERA_HEIGHT, CAMERA_WIDTH)
        assert img.dtype == torch.float32
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    # TODO(kartik): _rgb suffix stripping should be configurable
    def test_processor_strips_rgb_suffix(self, processor):
        """Test that _rgb suffix is stripped from camera names."""
        # Using smaller test dimensions (testing generic behavior)
        test_cam_size = 256
        obs = {
            f"{OBS_STR}.camera_obs": {
                "left_cam_rgb": torch.randint(
                    0, 255, (1, test_cam_size, test_cam_size, 3), dtype=torch.uint8
                ),
                "right_cam_rgb": torch.randint(
                    0, 255, (1, test_cam_size, test_cam_size, 3), dtype=torch.uint8
                ),
            },
        }

        processed = processor.observation(obs)

        assert f"{OBS_IMAGES}.left_cam" in processed
        assert f"{OBS_IMAGES}.right_cam" in processed
        assert f"{OBS_IMAGES}.left_cam_rgb" not in processed

    def test_processor_handles_single_image(self, processor):
        """Test processor handles single image without batch dimension."""
        obs = {
            f"{OBS_STR}.camera_obs": {
                "robot_pov_cam_rgb": torch.randint(
                    0, 255, (CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=torch.uint8
                ),
            },
        }

        processed = processor.observation(obs)

        img = processed[f"{OBS_IMAGES}.robot_pov_cam"]
        # Batch dimension added
        assert img.shape == (1, 3, CAMERA_HEIGHT, CAMERA_WIDTH)

    def test_processor_handles_float32_images(self, processor):
        """Test processor handles float32 images (already normalized)."""
        # Using smaller test dimensions for this generic behavior test
        test_cam_size = 256
        obs = {
            f"{OBS_STR}.camera_obs": {
                "robot_pov_cam_rgb": torch.rand(
                    BATCH_SIZE, test_cam_size, test_cam_size, 3, dtype=torch.float32
                ),
            },
        }

        processed = processor.observation(obs)

        img = processed[f"{OBS_IMAGES}.robot_pov_cam"]
        assert img.dtype == torch.float32

    def test_processor_concatenates_state_components(self, processor):
        """Test that state components are concatenated in order."""
        # Custom processor with multiple state keys
        processor_multi = IsaaclabArenaProcessorStep()
        processor_multi.state_keys = ("robot_joint_pos", "actions")

        obs = {
            f"{OBS_STR}.policy": {
                "robot_joint_pos": torch.ones(BATCH_SIZE, STATE_DIM),
                "actions": torch.ones(BATCH_SIZE, ACTION_DIM) * 2,
            },
        }

        processed = processor_multi.observation(obs)

        state = processed[OBS_STATE]
        expected_concat_dim = STATE_DIM + ACTION_DIM  # 54 + 36 = 90
        assert state.shape == (BATCH_SIZE, expected_concat_dim)
        # First STATE_DIM should be 1s, last ACTION_DIM should be 2s
        assert torch.all(state[:, :STATE_DIM] == 1)
        assert torch.all(state[:, STATE_DIM:] == 2)

    def test_processor_handles_1d_state(self, processor):
        """Test processor handles 1D state input."""
        obs = {
            f"{OBS_STR}.policy": {
                "robot_joint_pos": torch.randn(STATE_DIM),  # No batch dim
            },
        }

        processed = processor.observation(obs)

        assert processed[OBS_STATE].shape == (1, STATE_DIM)

    def test_processor_handles_higher_dim_state(self, processor):
        """Test processor flattens higher dimensional state."""
        # 6x9 = 54 = STATE_DIM (assuming STATE_DIM is 54)
        obs = {
            f"{OBS_STR}.policy": {
                "robot_joint_pos": torch.randn(BATCH_SIZE, 6, 9),  # 6x9 = 54
            },
        }

        processed = processor.observation(obs)

        assert processed[OBS_STATE].shape == (BATCH_SIZE, STATE_DIM)

    def test_processor_passes_through_other_keys(self, processor):
        """Test that other observation keys are passed through."""
        obs = {
            f"{OBS_STR}.policy": {
                "robot_joint_pos": torch.randn(BATCH_SIZE, STATE_DIM),
            },
            "task": ["Open microwave", "Open microwave"],
        }

        processed = processor.observation(obs)

        assert "task" in processed
        assert processed["task"] == ["Open microwave", "Open microwave"]

    def test_processor_in_pipeline(self, sample_observation):
        """Test processor works in a pipeline."""
        pipeline = PolicyProcessorPipeline(steps=[IsaaclabArenaProcessorStep()])

        processed = pipeline(sample_observation)

        assert OBS_STATE in processed
        assert f"{OBS_IMAGES}.robot_pov_cam" in processed

    def test_processor_handles_missing_camera_obs(self, processor):
        """Test processor handles observation without camera data."""
        obs = {
            f"{OBS_STR}.policy": {
                "robot_joint_pos": torch.randn(BATCH_SIZE, STATE_DIM),
            },
        }

        processed = processor.observation(obs)

        assert OBS_STATE in processed
        # No camera keys should be present
        assert not any(k.startswith(OBS_IMAGES) for k in processed)

    def test_processor_handles_missing_policy_obs(self, processor):
        """Test processor handles observation without policy data."""
        obs = {
            f"{OBS_STR}.camera_obs": {
                "robot_pov_cam_rgb": torch.randint(
                    0, 255, (BATCH_SIZE, CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=torch.uint8
                ),
            },
        }

        processed = processor.observation(obs)

        # Camera should still be processed
        assert f"{OBS_IMAGES}.robot_pov_cam" in processed
        # No state key
        assert OBS_STATE not in processed
