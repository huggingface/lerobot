"""FastAPI main application for LeRobot Web UI."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from lerobot.webui.backend.services.config_manager import ConfigManager
from lerobot.webui.backend.services.process_manager import process_manager

# Initialize FastAPI app
app = FastAPI(
    title="LeRobot Web UI",
    description="Modern web interface for xLeRobot teleoperation and data recording",
    version="1.0.0",
)

# CORS middleware for Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
config_manager = ConfigManager()

# Mount static directory for camera previews
repo_root = Path(__file__).parent.parent.parent.parent.parent
outputs_dir = repo_root / "outputs"
if not outputs_dir.exists():
    outputs_dir.mkdir(parents=True, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")


# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "LeRobot Web UI is running"}


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print("LeRobot Web UI starting...")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("LeRobot Web UI shutting down...")
    await process_manager.cleanup()


# Import and include routers
from lerobot.webui.backend.api import (
    calibration,
    config,
    huggingface,
    recording,
    setup,
    system,
    teleoperation,
)

app.include_router(setup.router, prefix="/api/setup", tags=["setup"])
app.include_router(calibration.router, prefix="/api/calibration", tags=["calibration"])
app.include_router(teleoperation.router, prefix="/api/teleoperation", tags=["teleoperation"])
app.include_router(recording.router, prefix="/api/recording", tags=["recording"])
app.include_router(config.router, prefix="/api/config", tags=["config"])
app.include_router(huggingface.router, prefix="/api/huggingface", tags=["huggingface"])
app.include_router(system.router, prefix="/api/system", tags=["system"])

# WebSocket endpoint
from lerobot.webui.backend.websockets.logs import router as websocket_router

app.include_router(websocket_router)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
