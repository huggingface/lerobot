"""Configuration models for the web UI."""

from typing import List, Optional

from pydantic import BaseModel, Field


class CameraConfig(BaseModel):
    """Camera configuration."""

    index: int = Field(..., description="Camera index or path")
    name: str = Field(..., description="Camera name (e.g., 'hand_cam', 'front_cam')")
    width: int = Field(640, description="Camera width in pixels")
    height: int = Field(480, description="Camera height in pixels")
    fps: int = Field(30, description="Camera frames per second")


class SingleArmConfig(BaseModel):
    """Configuration for single arm setup."""

    follower_port: Optional[str] = Field(None, description="Follower robot port")
    leader_port: Optional[str] = Field(None, description="Leader teleoperator port")
    follower_id: Optional[str] = Field(None, description="Follower calibration ID (filename without .json)")
    leader_id: Optional[str] = Field(None, description="Leader calibration ID (filename without .json)")
    cameras: List[CameraConfig] = Field(default_factory=list, description="Camera configurations")


class BimanualConfig(BaseModel):
    """Configuration for bimanual arm setup."""

    left_follower_port: Optional[str] = Field(None, description="Left follower robot port")
    left_leader_port: Optional[str] = Field(None, description="Left leader teleoperator port")
    right_follower_port: Optional[str] = Field(None, description="Right follower robot port")
    right_leader_port: Optional[str] = Field(None, description="Right leader teleoperator port")
    left_follower_id: Optional[str] = Field(None, description="Left follower calibration ID")
    left_leader_id: Optional[str] = Field(None, description="Left leader calibration ID")
    right_follower_id: Optional[str] = Field(None, description="Right follower calibration ID")
    right_leader_id: Optional[str] = Field(None, description="Right leader calibration ID")
    cameras: List[CameraConfig] = Field(default_factory=list, description="Camera configurations")


class LastRecordingConfig(BaseModel):
    """Last recording configuration."""

    repo_id: Optional[str] = Field(None, description="HuggingFace repo ID")
    task: Optional[str] = Field(None, description="Task description")
    num_episodes: int = Field(10, description="Number of episodes")
    episode_time_s: int = Field(100, description="Episode time in seconds")


class Config(BaseModel):
    """Main configuration model."""

    mode: str = Field("bimanual", description="Operation mode: 'single' or 'bimanual'")
    single_arm: SingleArmConfig = Field(default_factory=SingleArmConfig)
    bimanual: BimanualConfig = Field(default_factory=BimanualConfig)
    last_recording: LastRecordingConfig = Field(default_factory=LastRecordingConfig)
