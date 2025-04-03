#!/usr/bin/env python3
# Copyright 2023 Hugging Face Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import base64
import json
import os
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import flask
import numpy as np
import requests
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from PIL import Image

from lerobot.data.dataset import Dataset
from lerobot.data.episode import Episode
from lerobot.data.frame import Frame
from lerobot.data.utils import get_dataset_path

from allowed_hosts import ALLOWED_SCHEMES, ALLOWED_HOSTS

app = Flask(__name__)
CORS(app)


def validate_url(url):
    """Validate URL against allowed schemes and hosts."""
    parsed_url = urllib.parse.urlparse(url)
    
    # Check if scheme is allowed
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        return False
    
    # Check if host is allowed
    if parsed_url.netloc not in ALLOWED_HOSTS:
        return False
    
    return True


def get_episode_data(dataset_path: Path, episode_id: str) -> Tuple[Episode, List[Frame]]:
    dataset = Dataset(dataset_path)
    episode = dataset.get_episode(episode_id)
    frames = episode.get_frames()
    return episode, frames


def get_episode_metadata(episode: Episode, frames: List[Frame]) -> Dict:
    metadata = {
        "episode_id": episode.episode_id,
        "num_frames": len(frames),
        "actions": [],
    }

    for frame in frames:
        if frame.action is not None:
            metadata["actions"].append(
                {
                    "frame_id": frame.frame_id,
                    "action_type": frame.action.action_type,
                }
            )

    return metadata


def encode_image(image_path: Union[str, Path]) -> str:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def get_frame_data(frame: Frame) -> Dict:
    frame_data = {"frame_id": frame.frame_id}

    # Add RGB image
    if frame.rgb_path is not None:
        frame_data["rgb"] = encode_image(frame.rgb_path)

    # Add depth image
    if frame.depth_path is not None:
        # Convert depth image to color map for visualization
        depth_image = cv2.imread(str(frame.depth_path), cv2.IMREAD_ANYDEPTH)
        if depth_image is not None:
            # Normalize depth image to 0-255
            depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_image_normalized = depth_image_normalized.astype(np.uint8)
            # Apply color map
            depth_image_colormap = cv2.applyColorMap(depth_image_normalized, cv2.COLORMAP_JET)
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                cv2.imwrite(temp_file.name, depth_image_colormap)
                frame_data["depth"] = encode_image(temp_file.name)
            # Remove temporary file
            os.unlink(temp_file.name)

    # Add action
    if frame.action is not None:
        frame_data["action"] = {
            "action_type": frame.action.action_type,
            "action_args": frame.action.action_args,
        }

    # Add state
    if frame.state is not None:
        frame_data["state"] = frame.state

    return frame_data


@app.route("/")
def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dataset Viewer</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            h1 {
                color: #333;
            }
            .episode-selector {
                margin-bottom: 20px;
            }
            .frame-viewer {
                display: flex;
                flex-wrap: wrap;
            }
            .frame-container {
                margin-right: 20px;
                margin-bottom: 20px;
            }
            .frame-image {
                max-width: 400px;
                border: 1px solid #ddd;
            }
            .frame-info {
                margin-top: 10px;
                background-color: #f9f9f9;
                padding: 10px;
                border-radius: 3px;
                max-width: 400px;
            }
            .navigation {
                margin-top: 20px;
                display: flex;
                justify-content: space-between;
            }
            button {
                padding: 8px 16px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
            button:disabled {
                background-color: #cccccc;
                cursor: not-allowed;
            }
            .frame-counter {
                margin-top: 10px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Dataset Viewer</h1>
            
            <div class="episode-selector">
                <label for="episode-id">Episode ID:</label>
                <input type="text" id="episode-id" placeholder="Enter episode ID">
                <button onclick="loadEpisode()">Load Episode</button>
            </div>
            
            <div class="frame-counter">
                Frame: <span id="current-frame">0</span> / <span id="total-frames">0</span>
            </div>
            
            <div class="frame-viewer">
                <div class="frame-container">
                    <h3>RGB Image</h3>
                    <img id="rgb-image" class="frame-image" src="" alt="RGB Image">
                </div>
                
                <div class="frame-container">
                    <h3>Depth Image</h3>
                    <img id="depth-image" class="frame-image" src="" alt="Depth Image">
                </div>
            </div>
            
            <div class="frame-info" id="frame-info">
                <h3>Frame Information</h3>
                <pre id="frame-data"></pre>
            </div>
            
            <div class="navigation">
                <button id="prev-button" onclick="prevFrame()" disabled>Previous Frame</button>
                <button id="next-button" onclick="nextFrame()" disabled>Next Frame</button>
            </div>
        </div>
        
        <script>
            let currentEpisode = null;
            let currentFrameIndex = 0;
            let frames = [];
            
            function loadEpisode() {
                const episodeId = document.getElementById('episode-id').value;
                if (!episodeId) {
                    alert('Please enter an episode ID');
                    return;
                }
                
                fetch(`/api/episode/${episodeId}`)
                    .then(response => response.json())
                    .then(data => {
                        currentEpisode = data;
                        document.getElementById('total-frames').textContent = data.num_frames;
                        currentFrameIndex = 0;
                        loadFrame(0);
                        document.getElementById('prev-button').disabled = true;
                        document.getElementById('next-button').disabled = data.num_frames <= 1;
                    })
                    .catch(error => {
                        console.error('Error loading episode:', error);
                        alert('Error loading episode. Please check the episode ID and try again.');
                    });
            }
            
            function loadFrame(frameIndex) {
                if (!currentEpisode) return;
                
                fetch(`/api/episode/${currentEpisode.episode_id}/frame/${frameIndex}`)
                    .then(response => response.json())
                    .then(data => {
                        // Update RGB image
                        if (data.rgb) {
                            document.getElementById('rgb-image').src = `data:image/jpeg;base64,${data.rgb}`;
                        } else {
                            document.getElementById('rgb-image').src = '';
                        }
                        
                        // Update depth image
                        if (data.depth) {
                            document.getElementById('depth-image').src = `data:image/jpeg;base64,${data.depth}`;
                        } else {
                            document.getElementById('depth-image').src = '';
                        }
                        
                        // Update frame info
                        const frameInfo = {
                            frame_id: data.frame_id,
                            action: data.action,
                            state: data.state
                        };
                        document.getElementById('frame-data').textContent = JSON.stringify(frameInfo, null, 2);
                        
                        // Update current frame counter
                        document.getElementById('current-frame').textContent = frameIndex + 1;
                        
                        // Update navigation buttons
                        document.getElementById('prev-button').disabled = frameIndex === 0;
                        document.getElementById('next-button').disabled = frameIndex >= currentEpisode.num_frames - 1;
                    })
                    .catch(error => {
                        console.error('Error loading frame:', error);
                        alert('Error loading frame data.');
                    });
            }
            
            function prevFrame() {
                if (currentFrameIndex > 0) {
                    currentFrameIndex--;
                    loadFrame(currentFrameIndex);
                }
            }
            
            function nextFrame() {
                if (currentEpisode && currentFrameIndex < currentEpisode.num_frames - 1) {
                    currentFrameIndex++;
                    loadFrame(currentFrameIndex);
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content


@app.route("/api/episode/<episode_id>")
def get_episode(episode_id):
    dataset_path = request.args.get("dataset_path", None)
    if dataset_path is None:
        return jsonify({"error": "dataset_path parameter is required"}), 400

    try:
        episode, frames = get_episode_data(Path(dataset_path), episode_id)
        metadata = get_episode_metadata(episode, frames)
        return jsonify(metadata)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/episode/<episode_id>/frame/<int:frame_index>")
def get_frame(episode_id, frame_index):
    dataset_path = request.args.get("dataset_path", None)
    if dataset_path is None:
        return jsonify({"error": "dataset_path parameter is required"}), 400

    try:
        episode, frames = get_episode_data(Path(dataset_path), episode_id)
        if frame_index < 0 or frame_index >= len(frames):
            return jsonify({"error": f"Frame index {frame_index} out of range"}), 400

        frame_data = get_frame_data(frames[frame_index])
        return jsonify(frame_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/proxy")
def proxy():
    url = request.args.get("url", None)
    if url is None:
        return jsonify({"error": "url parameter is required"}), 400

    # Validate URL against allowed schemes and hosts
    if not validate_url(url):
        return jsonify({"error": "URL is not allowed"}), 403

    try:
        # Make the request but don't forward headers from the original request
        # to prevent header injection
        response = requests.get(url, timeout=5)
        
        # Don't return the actual response to the user, just a success message
        # This prevents SSRF attacks where the response might contain sensitive information
        return jsonify({
            "status": "success",
            "message": "Request completed successfully",
            "status_code": response.status_code
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset in HTML")
    parser.add_argument("--dataset-name", type=str, help="Name of the dataset")
    parser.add_argument("--dataset-path", type=str, help="Path to the dataset")
    parser.add_argument("--host", type=str, default="localhost", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()

    if args.dataset_name is not None:
        dataset_path = get_dataset_path(args.dataset_name)
    elif args.dataset_path is not None:
        dataset_path = Path(args.dataset_path)
    else:
        print("Either --dataset-name or --dataset-path must be provided")
        sys.exit(1)

    app.config["dataset_path"] = dataset_path
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
