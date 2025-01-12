import threading
import time
from pathlib import Path

import zmq
from flask import Flask, render_template
from flask_socketio import SocketIO

# Get template directory path
template_dir = Path(__file__).resolve().parent.parent / "templates"

# Initialize Flask with custom template directory
app = Flask(__name__, template_folder=str(template_dir))
socketio = SocketIO(app, cors_allowed_origins="*")

# Global dictionary to hold the latest data from ZeroMQ
latest_data = {
    "observation": {},
    "config": {}
}

zmq_context = zmq.Context()

# For receiving updates from ControlContext
subscriber_socket = zmq_context.socket(zmq.SUB)
subscriber_socket.connect("tcp://127.0.0.1:5555")
subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "")

# For sending keydown events to ControlContext
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
                    "log_items": message.get("log_items", []),
                    "countdown_time": message.get("countdown_time")
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

                # Update latest observation and config
                latest_data["observation"].update(processed_data)
                latest_data["config"].update(processed_data.get("config", {}))
                
                # # Emit the observation data to the browser
                socketio.emit("observation_update", processed_data)
        
                
            elif message.get("type") == "config_update":
                # Handle dedicated config updates
                config_data = message.get("config", {})
                latest_data["config"].update(config_data)
                
                # Emit configuration update to browser
                socketio.emit("config_update", {
                    "timestamp": message.get("timestamp"),
                    "config": config_data
                })
            elif message.get("type") == "log_say":
                data = message.get("message")
                timestamp = message.get("timestamp")
                socketio.emit("log_say", {
                    "timestamp": timestamp,
                    "message": data
                })
                
        except Exception as e:
            print(f"ZMQ consumer error: {e}")
            time.sleep(1)


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
    if latest_data["observation"]:
        socketio.emit("observation_update", latest_data["observation"])
    if latest_data["config"]:
        socketio.emit("config_update", {
            "timestamp": time.time(),
            "config": latest_data["config"]
        })

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