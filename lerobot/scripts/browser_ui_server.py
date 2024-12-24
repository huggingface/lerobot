from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import numpy as np
import base64
import json
import threading
import time
import zmq
from pathlib import Path

# Get template directory path
template_dir = Path(__file__).resolve().parent.parent / "templates"

# Initialize Flask with custom template directory
app = Flask(__name__, template_folder=str(template_dir))
socketio = SocketIO(app, cors_allowed_origins="*")

# Global dictionary to hold the latest observation data from ZeroMQ
latest_observation = {}

zmq_context = zmq.Context()

# For recieving observation (camera frames, state, events) from ControlContext
# so we can send them to the browser
subscriber_socket = zmq_context.socket(zmq.SUB)
subscriber_socket.connect("tcp://127.0.0.1:5555")
subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "")

# For sending keydown events from the browser to ControlContext
command_publisher = zmq_context.socket(zmq.PUB)
command_publisher.bind("tcp://127.0.0.1:5556")

def zmq_consumer():
    while True:
        try:
            message = subscriber_socket.recv_json()
            if message.get("type") == "observation_update":
                processed_data = {
                    "timestamp": message.get("timestamp"),
                    "images": {},
                    "state": {},
                    "events": message.get("events", {}),
                    "config": message.get("config", {}),
                    "log_items": message.get("log_items", [])
                }

                # Process observation data
                observation_data = message.get("data", {})
                for key, value in observation_data.items():
                    if "image" in key:
                        if value["type"] == "image":
                            processed_data["images"][key.split(".")[-1]] = value["data"]
                    else:
                        if value["type"] == "tensor":
                            processed_data["state"][key] = {
                                "data": value["data"],
                                "shape": value["shape"]
                            }

                # Update latest observation
                latest_observation.update(processed_data)
                
                # Emit the observation data to the browser
                socketio.emit("observation_update", processed_data)
                
        except Exception as e:
            print(f"ZMQ consumer error: {e}")
            time.sleep(1)
            
        time.sleep(0.001)  # Small sleep to prevent busy-waiting


@socketio.on("keydown_event")
def handle_keydown_event(data):
    """
    When the browser sends a keydown_event, we publish it over ZeroMQ.
    """
    key_pressed = data.get("key")

    # Publish over ZeroMQ
    message = {
        "type": "command",
        "command": "keydown",
        "key_pressed": key_pressed
    }
    command_publisher.send_json(message)

@app.route("/")
def index():
    """Render the main page."""
    return render_template("browser_ui.html")

@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    print("Client connected")
    # Send current state if available
    if latest_observation:
        socketio.emit("observation_update", latest_observation)

@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    print("Client disconnected")

def run_server(host="0.0.0.0", port=8000):
    """Run the Flask-SocketIO server."""
    # Start ZMQ consumer in a background thread
    zmq_thread = threading.Thread(target=zmq_consumer, daemon=True)
    zmq_thread.start()
    
    # Run Flask-SocketIO app
    socketio.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host IP address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()
    
    print(f"Starting server at {args.host}:{args.port}")
    run_server(host=args.host, port=args.port)