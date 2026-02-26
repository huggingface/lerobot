"""Calibration API endpoints."""

import asyncio
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from lerobot.webui.backend.models.system import CalibrationStatus
from lerobot.webui.backend.services.auto_calibration import AutoCalibrationService, CalibrationProgress
from lerobot.webui.backend.services.calibration_service import CalibrationService
from lerobot.webui.backend.services.config_manager import ConfigManager
from lerobot.webui.backend.services.process_manager import process_manager

router = APIRouter()
calibration_service = CalibrationService()
config_manager = ConfigManager()
auto_cal_service = AutoCalibrationService()

# Track active auto-calibration sessions
_active_sessions: Dict[str, asyncio.Task] = {}


class CalibrationStartRequest(BaseModel):
    """Request to start calibration."""

    device_type: str  # "robot" or "teleoperator"
    device_id: str
    robot_type: str
    port: str


class CalibrationStartResponse(BaseModel):
    """Response from starting calibration."""

    process_id: str
    message: str


@router.get("/files", response_model=List[str])
async def list_calibration_files(category: str, robot_type: str):
    """List available calibration files for a given category and robot type.

    Args:
        category: "robots" or "teleoperators".
        robot_type: Robot type (e.g., "so101_follower", "bi_so101_follower").
    """
    try:
        return calibration_service.list_calibration_files(category, robot_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list calibration files: {e}")


@router.get("/missing", response_model=List[CalibrationStatus])
async def get_missing_calibrations():
    """Get list of devices missing calibration based on current config."""
    try:
        config = config_manager.load_config()
        return calibration_service.list_missing_calibrations(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check calibrations: {e}")


@router.get("/status", response_model=List[CalibrationStatus])
async def get_calibration_status():
    """Get calibration status for all devices in current config."""
    try:
        config = config_manager.load_config()
        statuses = []

        if config.mode == "bimanual":
            # Check bimanual devices
            devices = [
                ("robot", "bimanual_follower", "bi_so101_follower", None),
                ("teleoperator", "bimanual_leader", "bi_so101_leader", None),
            ]
        else:
            # Check single arm devices
            devices = [
                ("robot", "single_follower", "so101_follower", config.single_arm.follower_port),
                ("teleoperator", "single_leader", "so101_leader", config.single_arm.leader_port),
            ]

        for device_type, device_id, robot_type, port in devices:
            if port or config.mode == "bimanual":
                status = calibration_service.check_calibration(device_type, device_id, robot_type, port)
                statuses.append(status)

        return statuses

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get calibration status: {e}")


@router.post("/start", response_model=CalibrationStartResponse)
async def start_calibration(request: CalibrationStartRequest):
    """Start calibration process for a device."""
    try:
        command = calibration_service.build_calibration_command(
            request.device_type, request.device_id, request.robot_type, request.port
        )

        process_id = await process_manager.start_process(command, "calibration")

        return CalibrationStartResponse(
            process_id=process_id, message=f"Calibration started for {request.device_id}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start calibration: {e}")


@router.post("/stop/{process_id}")
async def stop_calibration(process_id: str):
    """Stop calibration process."""
    try:
        success = await process_manager.stop_process(process_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Process {process_id} not found")

        return {"message": "Calibration stopped successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop calibration: {e}")


# --- Auto-calibration endpoints ---


class AutoCalibrationRequest(BaseModel):
    """Request to start auto-calibration for a motor."""

    port: str
    motor_name: str
    device_type: str = "robot"  # "robot" or "teleoperator"
    robot_type: str = "so101_follower"
    device_id: str = "left_follower"
    step_size: Optional[int] = None


class AutoCalibrationResult(BaseModel):
    """Result of auto-calibration."""

    session_id: str
    motor_name: str
    calibration: Optional[dict] = None
    saved_path: Optional[str] = None
    error: Optional[str] = None


@router.post("/auto/start", response_model=AutoCalibrationResult)
async def start_auto_calibration(request: AutoCalibrationRequest):
    """Start auto-calibration for a single motor.

    This directly controls the servo to find its physical limits.
    The motor will move slowly in each direction until it detects a stall.
    """
    session_id = str(uuid.uuid4())[:8]

    try:
        result = await auto_cal_service.auto_calibrate_motor(
            port=request.port,
            motor_name=request.motor_name,
            step_size=request.step_size,
        )

        if result.get("cancelled"):
            return AutoCalibrationResult(
                session_id=session_id,
                motor_name=request.motor_name,
                error="Calibration was cancelled",
            )

        # Save to file
        saved_path = auto_cal_service.save_calibration(
            calibration_data=result,
            device_type=request.device_type,
            robot_type=request.robot_type,
            device_id=request.device_id,
        )

        # Write to motor EEPROM
        auto_cal_service.write_calibration_to_motors(
            port=request.port,
            calibration_data=result,
        )

        return AutoCalibrationResult(
            session_id=session_id,
            motor_name=request.motor_name,
            calibration=result,
            saved_path=str(saved_path),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-calibration failed: {e}")


@router.post("/auto/cancel")
async def cancel_auto_calibration():
    """Cancel any running auto-calibration."""
    auto_cal_service.cancel()
    return {"message": "Cancellation requested"}


@router.websocket("/auto/ws")
async def auto_calibration_ws(websocket: WebSocket):
    """WebSocket endpoint for auto-calibration with real-time progress.

    Send a JSON message to start:
    {
        "port": "/dev/tty.usbmodemXXX",
        "motor_name": "gripper",
        "device_type": "robot",
        "robot_type": "so101_follower",
        "device_id": "left_follower"
    }

    Receives JSON progress messages:
    {
        "type": "progress",
        "motor_name": "gripper",
        "phase": "finding_min",
        "current_position": 1234,
        "range_min": null,
        "range_max": null,
        "progress_pct": 25.0
    }

    Final result message:
    {
        "type": "result",
        "calibration": {...},
        "saved_path": "/path/to/file.json"
    }
    """
    await websocket.accept()

    try:
        # Wait for start command
        data = await websocket.receive_json()
        port = data["port"]
        motor_name = data["motor_name"]
        device_type = data.get("device_type", "robot")
        robot_type = data.get("robot_type", "so101_follower")
        device_id = data.get("device_id", "left_follower")
        step_size = data.get("step_size")

        # Progress callback that sends updates over WebSocket
        async def send_progress(progress: CalibrationProgress):
            try:
                await websocket.send_json({
                    "type": "progress",
                    "motor_name": progress.motor_name,
                    "phase": progress.phase,
                    "current_position": progress.current_position,
                    "range_min": progress.range_min,
                    "range_max": progress.range_max,
                    "progress_pct": progress.progress_pct,
                })
            except Exception:
                pass

        # We need a sync-compatible wrapper since the motor bus is synchronous
        # but we want async WebSocket updates
        progress_queue: asyncio.Queue[CalibrationProgress] = asyncio.Queue()

        def on_progress(progress: CalibrationProgress):
            progress_queue.put_nowait(progress)

        # Run calibration in a background task
        cal_task = asyncio.create_task(
            auto_cal_service.auto_calibrate_motor(
                port=port,
                motor_name=motor_name,
                on_progress=on_progress,
                step_size=step_size,
            )
        )

        # Forward progress updates while calibration runs
        while not cal_task.done():
            try:
                progress = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                await send_progress(progress)
            except asyncio.TimeoutError:
                # Check if client sent a cancel message
                try:
                    msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.01)
                    if msg.get("action") == "cancel":
                        auto_cal_service.cancel()
                        await websocket.send_json({"type": "cancelled"})
                except (asyncio.TimeoutError, WebSocketDisconnect):
                    pass

        # Drain remaining progress messages
        while not progress_queue.empty():
            progress = progress_queue.get_nowait()
            await send_progress(progress)

        # Get result
        result = await cal_task

        if result.get("cancelled"):
            await websocket.send_json({"type": "cancelled"})
        else:
            # Save calibration
            saved_path = auto_cal_service.save_calibration(
                calibration_data=result,
                device_type=device_type,
                robot_type=robot_type,
                device_id=device_id,
            )

            # Write to motor EEPROM
            auto_cal_service.write_calibration_to_motors(
                port=port,
                calibration_data=result,
            )

            await websocket.send_json({
                "type": "result",
                "calibration": result,
                "saved_path": str(saved_path),
            })

    except WebSocketDisconnect:
        auto_cal_service.cancel()
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
