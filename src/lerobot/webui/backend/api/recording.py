"""Recording API endpoints."""

import json
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException

from lerobot.webui.backend.models.recording import RecordingRequest, RecordingResponse
from lerobot.webui.backend.models.system import ProcessStatus
from lerobot.webui.backend.services.config_manager import ConfigManager
from lerobot.webui.backend.services.process_manager import process_manager

router = APIRouter()
config_manager = ConfigManager()


def build_recording_command(config, request: RecordingRequest) -> list[str]:
    """Build recording command from config and request."""
    # Build cameras JSON
    cameras_dict = {}
    if config.mode == "bimanual":
        for cam in config.bimanual.cameras:
            cameras_dict[cam.name] = {
                "type": "opencv",
                "index_or_path": cam.index,
                "width": cam.width,
                "height": cam.height,
                "fps": cam.fps,
            }

        return [
            "lerobot-record",
            "--robot.type=bi_so101_follower",
            f"--robot.left_arm_port={config.bimanual.left_follower_port}",
            f"--robot.right_arm_port={config.bimanual.right_follower_port}",
            "--robot.id=bimanual_follower",
            f"--robot.cameras={json.dumps(cameras_dict)}",
            "--teleop.type=bi_so101_leader",
            f"--teleop.left_arm_port={config.bimanual.left_leader_port}",
            f"--teleop.right_arm_port={config.bimanual.right_leader_port}",
            "--teleop.id=bimanual_leader",
            f"--dataset.repo_id={request.repo_id}",
            f"--dataset.single_task={request.single_task}",
            f"--dataset.num_episodes={request.num_episodes}",
            f"--dataset.episode_time_s={request.episode_time_s}",
            f"--display_data={str(request.display_data).lower()}",
        ]
    else:
        for cam in config.single_arm.cameras:
            cameras_dict[cam.name] = {
                "type": "opencv",
                "index_or_path": cam.index,
                "width": cam.width,
                "height": cam.height,
                "fps": cam.fps,
            }

        return [
            "lerobot-record",
            "--robot.type=so101_follower",
            f"--robot.port={config.single_arm.follower_port}",
            "--robot.id=single_follower",
            f"--robot.cameras={json.dumps(cameras_dict)}",
            "--teleop.type=so101_leader",
            f"--teleop.port={config.single_arm.leader_port}",
            "--teleop.id=single_leader",
            f"--dataset.repo_id={request.repo_id}",
            f"--dataset.single_task={request.single_task}",
            f"--dataset.num_episodes={request.num_episodes}",
            f"--dataset.episode_time_s={request.episode_time_s}",
            f"--display_data={str(request.display_data).lower()}",
        ]


@router.post("/start", response_model=RecordingResponse)
async def start_recording(request: RecordingRequest):
    """Start dataset recording."""
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

            if not config.bimanual.cameras:
                raise HTTPException(status_code=400, detail="No cameras configured for recording")

        else:
            if not all([config.single_arm.follower_port, config.single_arm.leader_port]):
                raise HTTPException(
                    status_code=400, detail="Single arm mode requires both follower and leader ports"
                )

            if not config.single_arm.cameras:
                raise HTTPException(status_code=400, detail="No cameras configured for recording")

        # Build and start command
        command = build_recording_command(config, request)
        process_id = await process_manager.start_process(command, "recording")

        # Update last recording config
        config.last_recording.repo_id = request.repo_id
        config.last_recording.task = request.single_task
        config.last_recording.num_episodes = request.num_episodes
        config.last_recording.episode_time_s = request.episode_time_s
        config_manager.save_config(config)

        return RecordingResponse(process_id=process_id, message="Recording started successfully")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start recording: {e}")


@router.post("/stop/{process_id}")
async def stop_recording(process_id: str):
    """Stop recording."""
    try:
        success = await process_manager.stop_process(process_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Process {process_id} not found")

        return {"message": "Recording stopped successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop recording: {e}")


@router.get("/status/{process_id}", response_model=ProcessStatus)
async def get_recording_status(process_id: str):
    """Get recording process status."""
    try:
        status = await process_manager.get_status(process_id)

        if not status:
            raise HTTPException(status_code=404, detail=f"Process {process_id} not found")

        return status

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {e}")


@router.delete("/cache")
async def clear_cache(repo_id: str):
    """Clear dataset cache for a specific repo.

    Args:
        repo_id: HuggingFace repo ID (username/dataset_name).
    """
    try:
        cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id

        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            return {"message": f"Cache cleared for {repo_id}"}
        else:
            return {"message": f"No cache found for {repo_id}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {e}")
