"""Recording-related models."""

from typing import Optional

from pydantic import BaseModel, Field


class RecordingRequest(BaseModel):
    """Recording start request."""

    repo_id: str = Field(..., description="HuggingFace repo ID (username/dataset_name)")
    single_task: str = Field(..., description="Task description")
    num_episodes: int = Field(10, description="Number of episodes to record")
    episode_time_s: int = Field(100, description="Episode duration in seconds")
    reset_time_s: int = Field(10, description="Environment reset time between episodes in seconds")
    display_data: bool = Field(True, description="Whether to show Rerun visualization")


class RecordingResponse(BaseModel):
    """Recording start response."""

    process_id: str = Field(..., description="Process identifier for tracking")
    message: str = Field(..., description="Status message")


class HFRepoInfo(BaseModel):
    """HuggingFace repository information."""

    repo_id: str = Field(..., description="Repository ID (username/repo_name)")
    repo_type: str = Field("dataset", description="Repository type")
    private: bool = Field(False, description="Whether repository is private")
    url: Optional[str] = Field(None, description="Repository URL")


class CreateRepoRequest(BaseModel):
    """Request to create new HuggingFace repository."""

    repo_name: str = Field(..., description="Repository name (without username prefix)")
    private: bool = Field(False, description="Whether to create private repository")
