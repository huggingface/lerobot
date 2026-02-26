"""Setup-related models for ports and cameras."""

from typing import List, Optional

from pydantic import BaseModel, Field


class PortInfo(BaseModel):
    """Serial port information."""

    port: str = Field(..., description="Port path (e.g., '/dev/tty.usbmodem...')")
    description: Optional[str] = Field(None, description="Port description")
    hwid: Optional[str] = Field(None, description="Hardware ID")


class CameraInfo(BaseModel):
    """Camera information."""

    index: int = Field(..., description="Camera index")
    name: Optional[str] = Field(None, description="Camera name if available")
    backend: str = Field(..., description="Camera backend (opencv, realsense)")
    is_builtin: bool = Field(False, description="Whether this is a built-in camera")


class CameraPreview(BaseModel):
    """Camera preview with captured image."""

    index: int = Field(..., description="Camera index")
    image_path: str = Field(..., description="Path to captured preview image")
    image_url: str = Field(..., description="URL to access preview image")
