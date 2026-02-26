"""System status API endpoints."""

from fastapi import APIRouter, HTTPException

from lerobot.webui.backend.models.system import SystemStatus
from lerobot.webui.backend.services.calibration_service import CalibrationService
from lerobot.webui.backend.services.config_manager import ConfigManager
from lerobot.webui.backend.services.hf_service import HuggingFaceService
from lerobot.webui.backend.services.process_manager import process_manager

router = APIRouter()
config_manager = ConfigManager()
calibration_service = CalibrationService()
hf_service = HuggingFaceService()


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get overall system status."""
    try:
        # Get active processes
        active_processes_dict = await process_manager.get_active_processes()
        active_processes = list(active_processes_dict.values())

        # Get HF login status
        hf_status = hf_service.check_login()

        # Get missing calibrations
        config = config_manager.load_config()
        missing_calibrations = calibration_service.list_missing_calibrations(config)

        return SystemStatus(
            active_processes=active_processes, hf_status=hf_status, missing_calibrations=missing_calibrations
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {e}")
