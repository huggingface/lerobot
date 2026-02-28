"""Teleoperation API endpoints."""

import json

from fastapi import APIRouter, HTTPException

from lerobot.webui.backend.models.system import ProcessStatus
from lerobot.webui.backend.models.teleoperation import TeleoperationRequest, TeleoperationResponse
from lerobot.webui.backend.services.config_manager import ConfigManager
from lerobot.webui.backend.services.process_manager import process_manager

router = APIRouter()
config_manager = ConfigManager()


def build_teleoperation_command(config, display_data: bool = True) -> list[str]:
    """Build teleoperation command from config."""
    if config.mode == "bimanual":
        bi = config.bimanual
        return [
            "lerobot-teleoperate",
            "--robot.type=bi_so101_follower",
            f"--robot.left_arm_port={bi.left_follower_port}",
            f"--robot.right_arm_port={bi.right_follower_port}",
            f"--robot.id={bi.left_follower_id or 'bimanual_follower'}",
            "--teleop.type=bi_so101_leader",
            f"--teleop.left_arm_port={bi.left_leader_port}",
            f"--teleop.right_arm_port={bi.right_leader_port}",
            f"--teleop.id={bi.left_leader_id or 'bimanual_leader'}",
            f"--display_data={str(display_data).lower()}",
        ]
    else:
        sa = config.single_arm
        return [
            "lerobot-teleoperate",
            "--robot.type=so101_follower",
            f"--robot.port={sa.follower_port}",
            f"--robot.id={sa.follower_id or 'single_follower'}",
            "--teleop.type=so101_leader",
            f"--teleop.port={sa.leader_port}",
            f"--teleop.id={sa.leader_id or 'single_leader'}",
            f"--display_data={str(display_data).lower()}",
        ]


@router.post("/start", response_model=TeleoperationResponse)
async def start_teleoperation(request: TeleoperationRequest):
    """Start teleoperation."""
    try:
        config = config_manager.load_config()

        # Validate config
        if config.mode == "bimanual":
            if not all(
                [
                    config.bimanual.left_follower_port,
                    config.bimanual.left_leader_port,
                    config.bimanual.right_follower_port,
                    config.bimanual.right_leader_port,
                ]
            ):
                raise HTTPException(
                    status_code=400, detail="Bimanual mode requires all four ports to be configured"
                )
        else:
            if not all([config.single_arm.follower_port, config.single_arm.leader_port]):
                raise HTTPException(
                    status_code=400, detail="Single arm mode requires both follower and leader ports"
                )

        # Always enable display_data so motor positions are printed to stdout
        # (the WebUI parses them for the live motor panel). Suppress the Rerun
        # viewer by setting RERUN_ENABLED=false — rr.spawn() becomes a no-op.
        command = build_teleoperation_command(config, display_data=True)
        process_id = await process_manager.start_process(
            command, "teleoperation", env={"RERUN": "off"}
        )

        return TeleoperationResponse(process_id=process_id, message="Teleoperation started successfully")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start teleoperation: {e}")


@router.post("/stop/{process_id}")
async def stop_teleoperation(process_id: str):
    """Stop teleoperation."""
    try:
        success = await process_manager.stop_process(process_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Process {process_id} not found")

        return {"message": "Teleoperation stopped successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop teleoperation: {e}")


@router.get("/status/{process_id}", response_model=ProcessStatus)
async def get_teleoperation_status(process_id: str):
    """Get teleoperation process status."""
    try:
        status = await process_manager.get_status(process_id)

        if not status:
            raise HTTPException(status_code=404, detail=f"Process {process_id} not found")

        return status

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {e}")
