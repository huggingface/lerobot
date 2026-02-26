"""WebSocket endpoint for real-time log streaming."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from lerobot.webui.backend.services.process_manager import process_manager

router = APIRouter()


@router.websocket("/ws/logs/{process_id}")
async def websocket_logs(websocket: WebSocket, process_id: str):
    """Stream logs from a process via WebSocket.

    Args:
        websocket: WebSocket connection.
        process_id: Process identifier.
    """
    await websocket.accept()

    try:
        async for log_line in process_manager.stream_logs(process_id):
            await websocket.send_text(log_line)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(f"ERROR: {e}")
        await websocket.close()
