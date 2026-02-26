"""HuggingFace API endpoints."""

from typing import List

from fastapi import APIRouter, HTTPException

from lerobot.webui.backend.models.recording import CreateRepoRequest, HFRepoInfo
from lerobot.webui.backend.models.system import HFLoginStatus
from lerobot.webui.backend.services.hf_service import HuggingFaceService

router = APIRouter()
hf_service = HuggingFaceService()


@router.get("/whoami", response_model=HFLoginStatus)
async def check_login():
    """Check HuggingFace login status."""
    try:
        return hf_service.check_login()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check login: {e}")


@router.get("/repos", response_model=List[HFRepoInfo])
async def list_repos():
    """List user's dataset repositories."""
    try:
        login_status = hf_service.check_login()

        if not login_status.is_logged_in:
            raise HTTPException(status_code=401, detail="Not logged in to HuggingFace")

        return hf_service.list_repos(login_status.username)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list repos: {e}")


@router.post("/repos", response_model=HFRepoInfo)
async def create_repo(request: CreateRepoRequest):
    """Create a new dataset repository."""
    try:
        login_status = hf_service.check_login()

        if not login_status.is_logged_in:
            raise HTTPException(status_code=401, detail="Not logged in to HuggingFace")

        repo_info = hf_service.create_repo(login_status.username, request.repo_name, request.private)

        if not repo_info:
            raise HTTPException(status_code=500, detail="Failed to create repository")

        return repo_info

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create repo: {e}")
