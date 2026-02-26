"""System status and process models."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ProcessState(str, Enum):
    """Process state enum."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class ProcessStatus(BaseModel):
    """Process status model."""

    process_id: str = Field(..., description="Unique process identifier")
    process_type: str = Field(..., description="Process type (teleoperation, recording, calibration)")
    state: ProcessState = Field(..., description="Current process state")
    pid: Optional[int] = Field(None, description="Process ID if running")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    stopped_at: Optional[datetime] = Field(None, description="Stop timestamp")
    uptime_seconds: Optional[float] = Field(None, description="Uptime in seconds if running")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class CalibrationStatus(BaseModel):
    """Calibration status for a device."""

    device_type: str = Field(..., description="Device type (robot or teleoperator)")
    device_id: str = Field(..., description="Device identifier")
    robot_type: str = Field(..., description="Robot/teleoperator type (e.g., 'so101_follower')")
    port: Optional[str] = Field(None, description="Device port")
    is_calibrated: bool = Field(..., description="Whether device is calibrated")
    calibration_path: Optional[str] = Field(None, description="Path to calibration file if exists")


class HFLoginStatus(BaseModel):
    """HuggingFace login status."""

    is_logged_in: bool = Field(..., description="Whether user is logged in")
    username: Optional[str] = Field(None, description="HuggingFace username if logged in")


class SystemStatus(BaseModel):
    """Overall system status."""

    active_processes: list[ProcessStatus] = Field(default_factory=list, description="Active processes")
    hf_status: HFLoginStatus = Field(..., description="HuggingFace login status")
    missing_calibrations: list[CalibrationStatus] = Field(
        default_factory=list, description="Devices missing calibration"
    )
