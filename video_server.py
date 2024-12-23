from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
import base64
from starlette.websockets import WebSocketDisconnect
import zmq
import zmq.asyncio

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
        <div class="camera-container" id="cameras"></div>
        <script>
            let ws = new WebSocket(`ws://${window.location.host}/ws`);
            let cameras = {};

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                // data is an object: { cameraName: base64_jpeg, ... }

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

# Global dictionary to hold the latest frames from ZeroMQ
latest_frames = {}

# Set up a ZMQ context and SUB socket in asyncio mode
zmq_context = zmq.asyncio.Context()
subscriber_socket = zmq_context.socket(zmq.SUB)

# Connect to the producer's PUB socket
# Make sure this matches the IP/port from control_context.py
subscriber_socket.connect("tcp://127.0.0.1:5555")
subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all messages


async def zmq_consumer():
    """
    Continuously receive messages from ZeroMQ and update the global `latest_frames`.
    """
    while True:
        try:
            message = await subscriber_socket.recv_json()
            # message should look like: {"type": "frame_update", "frames": {cameraName: base64_jpeg, ...}}
            if message.get("type") == "frame_update":
                frames = message.get("frames", {})
                # Update the global dictionary
                for camera_name, b64_jpeg in frames.items():
                    latest_frames[camera_name] = b64_jpeg
        except Exception as e:
            print(f"ZMQ consumer error: {e}")
            await asyncio.sleep(1)
        # Small pause to avoid busy-loop
        await asyncio.sleep(0.001)


@app.on_event("startup")
async def startup_event():
    """
    When FastAPI starts, launch the background ZMQ consumer task.
    """
    asyncio.create_task(zmq_consumer())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send the latest frames to the websocket client
            if latest_frames:
                await websocket.send_json(latest_frames)
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
