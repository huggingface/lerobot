from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
from multiprocessing import shared_memory
import numpy as np
import json
import asyncio
import cv2
import base64
from starlette.websockets import WebSocketDisconnect

app = FastAPI()

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Camera Streams</title>
        <style>
            .camera-container {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                padding: 20px;
            }
            .camera-feed {
                margin: 10px;
            }
            canvas {
                border: 1px solid #ccc;
            }
        </style>
    </head>
    <body>
        <div class="camera-container" id="cameras">
        </div>
        <script>
            let ws = new WebSocket(`ws://${window.location.host}/ws`);
            let cameras = {};

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                // Create or update camera displays
                for (const [name, imageData] of Object.entries(data)) {
                    if (!cameras[name]) {
                        // Create new camera display
                        const container = document.createElement('div');
                        container.className = 'camera-feed';
                        
                        const title = document.createElement('h3');
                        title.textContent = name;
                        
                        const canvas = document.createElement('canvas');
                        canvas.id = `canvas-${name}`;
                        
                        container.appendChild(title);
                        container.appendChild(canvas);
                        document.getElementById('cameras').appendChild(container);
                        
                        cameras[name] = canvas;
                    }
                    
                    // Update image
                    const canvas = cameras[name];
                    const ctx = canvas.getContext('2d');
                    const img = new Image();
                    
                    img.onload = function() {
                        canvas.width = img.width;
                        canvas.height = img.height;
                        ctx.drawImage(img, 0, 0);
                    };
                    
                    img.src = 'data:image/jpeg;base64,' + imageData;
                }
            };
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

async def read_shared_memory():
    try:
        # Access metadata shared memory
        shm_metadata = shared_memory.SharedMemory(name='camera_metadata')
        metadata_str = shm_metadata.buf.tobytes().decode().split('\x00')[0]
        metadata = json.loads(metadata_str)

        frames = {}
        for name, info in metadata.items():
            try:
                # Access camera frame shared memory
                shm = shared_memory.SharedMemory(name=info['shm_name'])
                
                # Reconstruct numpy array from shared memory
                shape = tuple(info['shape'])
                dtype = np.dtype(info['dtype'])
                frame = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                
                # Convert to JPEG
                success, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if success:
                    frames[name] = base64.b64encode(buffer).decode('utf-8')
                
                shm.close()  # Close the shared memory access
            except FileNotFoundError:
                print(f"Shared memory for camera {name} not found. Skipping.")
                continue

        shm_metadata.close()  # Close metadata shared memory access
        return frames
    
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Error reading shared memory: {e}")
        return {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            frames = await read_shared_memory()
            if frames:
                await websocket.send_json(frames)
            await asyncio.sleep(0.033)  # ~30 FPS
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if websocket.client_state == "CONNECTED":
            await websocket.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0', help='Host IP address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    args = parser.parse_args()
    
    print(f"Starting server at {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
