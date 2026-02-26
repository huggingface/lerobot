"""Teleoperation-related models."""

from pydantic import BaseModel, Field


class TeleoperationRequest(BaseModel):
    """Teleoperation start request."""

    display_data: bool = Field(True, description="Whether to show Rerun visualization")


class TeleoperationResponse(BaseModel):
    """Teleoperation start response."""

    process_id: str = Field(..., description="Process identifier for tracking")
    message: str = Field(..., description="Status message")
