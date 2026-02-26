"""Configuration API endpoints."""

from fastapi import APIRouter, HTTPException

from lerobot.webui.backend.models.config import Config
from lerobot.webui.backend.services.config_manager import ConfigManager

router = APIRouter()
config_manager = ConfigManager()


@router.get("/", response_model=Config)
async def get_config():
    """Get current configuration."""
    return config_manager.load_config()


@router.post("/", response_model=Config)
async def save_config(config: Config):
    """Save configuration."""
    try:
        config_manager.save_config(config)
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {e}")


@router.delete("/", response_model=Config)
async def reset_config():
    """Reset configuration to defaults."""
    try:
        return config_manager.reset_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset config: {e}")
