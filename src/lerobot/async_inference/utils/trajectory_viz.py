# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Real-time trajectory visualization server for RTC inpainting assessment.

This module provides a WebSocket + HTTP server that streams action chunk data
to a browser-based visualization. It shows per-motor trajectories from up to
10 previous action chunks, with each chunk in a different color.

Usage:
    # Start standalone server (connects to robot client via shared queue):
    python -m lerobot.async_inference.utils.trajectory_viz --http_port 8088 --ws_port 8089

    # Then open http://localhost:8088 in your browser
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import time
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .metrics import EvActionChunk, EvExecutedAction

logger = logging.getLogger(__name__)


# =============================================================================
# WebSocket Server (using websockets library)
# =============================================================================


class TrajectoryVizServer:
    """WebSocket server that broadcasts action chunk data to connected clients."""

    def __init__(self, ws_port: int = 8089, http_port: int = 8088):
        self.ws_port = ws_port
        self.http_port = http_port
        self._chunk_queue: Queue[dict] = Queue(maxsize=100)
        self._clients: set = set()
        self._shutdown = threading.Event()

    def on_chunk(self, event: "EvActionChunk") -> None:
        """Callback to forward action chunks."""
        chunk_data = {
            "type": "action_chunk",
            "source_step": event.src_control_step,
            "actions": event.actions,
            "frozen_len": event.frozen_len,
            "timestamp": event.timestamp,
            # RTC visualization fields (may be None)
            "rtc_params": event.rtc_params,
            "prefix_weights": event.prefix_weights,
        }
        try:
            self._chunk_queue.put_nowait(chunk_data)
        except Full:
            # Drop oldest if full
            try:
                self._chunk_queue.get_nowait()
                self._chunk_queue.put_nowait(chunk_data)
            except Empty:
                pass

    async def _handler(self, websocket):
        """Handle a WebSocket connection."""
        self._clients.add(websocket)
        try:
            async for message in websocket:
                # Process incoming messages from clients (e.g., TrajectoryVizClient)
                # and queue them for broadcasting to all other clients (browsers)
                try:
                    data = json.loads(message)
                    # Queue for broadcasting (e.g., executed_action from robot client)
                    try:
                        self._chunk_queue.put_nowait(data)
                    except Full:
                        # Drop oldest if full
                        try:
                            self._chunk_queue.get_nowait()
                            self._chunk_queue.put_nowait(data)
                        except Empty:
                            pass
                except json.JSONDecodeError:
                    pass
        finally:
            self._clients.discard(websocket)

    async def _broadcaster(self):
        """Background task that broadcasts chunks to all connected clients."""
        while not self._shutdown.is_set():
            try:
                # Non-blocking check with small sleep
                await asyncio.sleep(0.01)
                try:
                    chunk_data = self._chunk_queue.get_nowait()
                except Empty:
                    continue

                if self._clients:
                    message = json.dumps(chunk_data)
                    # Broadcast to all connected clients
                    await asyncio.gather(
                        *[client.send(message) for client in self._clients],
                        return_exceptions=True,
                    )
            except Exception as e:
                logger.debug(f"Broadcaster error: {e}")

    async def _run_websocket_server(self):
        """Run the WebSocket server."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed. Run: uv pip install websockets")
            return

        async with websockets.serve(self._handler, "0.0.0.0", self.ws_port):
            logger.info(f"WebSocket server started on ws://0.0.0.0:{self.ws_port}")
            broadcaster_task = asyncio.create_task(self._broadcaster())
            try:
                await asyncio.Future()  # Run forever
            finally:
                broadcaster_task.cancel()

    def _run_http_server(self):
        """Run the HTTP server for static files."""
        # Get the directory containing the HTML file
        viz_dir = Path(__file__).parent

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(viz_dir), **kwargs)

            def log_message(self, format, *args):
                # Suppress access logs
                pass

            def do_GET(self):
                # Serve trajectory_viz.html as index
                if self.path == "/" or self.path == "/index.html":
                    self.path = "/trajectory_viz.html"
                return super().do_GET()

        server = HTTPServer(("0.0.0.0", self.http_port), Handler)
        logger.info(f"HTTP server started on http://0.0.0.0:{self.http_port}")
        server.serve_forever()

    def start(self):
        """Start both HTTP and WebSocket servers."""
        # Start HTTP server in a thread
        http_thread = threading.Thread(target=self._run_http_server, daemon=True)
        http_thread.start()

        # Run WebSocket server in the main thread's event loop
        asyncio.run(self._run_websocket_server())

    def stop(self):
        """Signal shutdown."""
        self._shutdown.set()


# =============================================================================
# WebSocket Client (for sending data to external visualization server)
# =============================================================================


class TrajectoryVizClient:
    """WebSocket client that sends action chunk data to a visualization server."""

    # Rate limit for "not connected" warnings (seconds between warnings)
    _NOT_CONNECTED_WARN_INTERVAL = 5.0

    def __init__(self, ws_url: str = "ws://localhost:8089"):
        self.ws_url = ws_url
        self._ws = None
        self._loop = None
        self._thread: threading.Thread | None = None
        self._queue: Queue[dict] = Queue(maxsize=100)
        self._shutdown = threading.Event()
        self._connected = False
        self._connection_attempted = False
        self._last_not_connected_warn: float = 0.0
        self._dropped_while_disconnected: int = 0

    def start(self) -> None:
        """Start the WebSocket client in a background thread."""
        logger.info(f"Starting trajectory viz client, will connect to {self.ws_url}")
        self._thread = threading.Thread(target=self._run, daemon=True, name="trajectory_viz_client")
        self._thread.start()

    def stop(self) -> None:
        """Stop the WebSocket client."""
        self._shutdown.set()
        if self._dropped_while_disconnected > 0:
            logger.warning(
                f"Trajectory viz client stopped. Total chunks dropped while disconnected: "
                f"{self._dropped_while_disconnected}"
            )

    def _run(self) -> None:
        """Run the WebSocket client event loop."""
        import asyncio

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_and_send())

    async def _connect_and_send(self) -> None:
        """Connect to the server and send queued data."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed. Run: uv pip install websockets")
            return

        while not self._shutdown.is_set():
            self._connection_attempted = True
            try:
                logger.info(f"Attempting to connect to visualization server at {self.ws_url}...")
                async with websockets.connect(self.ws_url) as ws:
                    self._connected = True
                    if self._dropped_while_disconnected > 0:
                        logger.info(
                            f"Connected to visualization server at {self.ws_url} "
                            f"(dropped {self._dropped_while_disconnected} chunks while disconnected)"
                        )
                        self._dropped_while_disconnected = 0
                    else:
                        logger.info(f"Connected to visualization server at {self.ws_url}")

                    while not self._shutdown.is_set():
                        try:
                            # Non-blocking check for data
                            await asyncio.sleep(0.01)
                            try:
                                chunk_data = self._queue.get_nowait()
                                await ws.send(json.dumps(chunk_data))
                            except Empty:
                                continue
                        except Exception as e:
                            logger.warning(f"WebSocket send error: {e}")
                            break

            except Exception as e:
                was_connected = self._connected
                self._connected = False
                if was_connected:
                    logger.warning(f"Disconnected from visualization server: {e}")
                else:
                    logger.warning(
                        f"Failed to connect to visualization server at {self.ws_url}: {e}. "
                        f"Make sure the server is running: python -m lerobot.async_inference.utils.trajectory_viz"
                    )
                await asyncio.sleep(2.0)

    def on_chunk(self, event: "EvActionChunk") -> None:
        """Callback to queue an action chunk for sending."""
        if not self._connected:
            self._dropped_while_disconnected += 1
            # Rate-limited warning to avoid log spam
            now = time.time()
            if now - self._last_not_connected_warn > self._NOT_CONNECTED_WARN_INTERVAL:
                self._last_not_connected_warn = now
                if self._connection_attempted:
                    logger.warning(
                        f"Trajectory viz: not connected to server, dropping chunks "
                        f"(total dropped: {self._dropped_while_disconnected}). "
                        f"Is the viz server running at {self.ws_url}?"
                    )
                else:
                    logger.debug("Trajectory viz: waiting for connection to establish...")
            return

        chunk_data = {
            "type": "action_chunk",
            "source_step": event.src_control_step,
            "actions": event.actions,
            "frozen_len": event.frozen_len,
            "timestamp": event.timestamp,
            # RTC visualization fields (may be None)
            "rtc_params": event.rtc_params,
            "prefix_weights": event.prefix_weights,
        }
        try:
            self._queue.put_nowait(chunk_data)
        except Full:
            # Drop oldest if full
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(chunk_data)
            except Empty:
                pass

    def on_executed_action(self, event: "EvExecutedAction") -> None:
        """Callback to queue an executed action for sending."""
        if not self._connected:
            # Don't count/warn for executed actions, just silently drop
            return

        action_data = {
            "type": "executed_action",
            "step": event.step,
            "action": event.action,
            "timestamp": event.timestamp,
        }
        try:
            self._queue.put_nowait(action_data)
        except Full:
            # Drop oldest if full
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(action_data)
            except Empty:
                pass


# =============================================================================
# Standalone Mode (for testing without robot client)
# =============================================================================


def generate_mock_chunks(server: TrajectoryVizServer, interval: float = 0.5):
    """Generate mock action chunks for testing the visualization."""
    import random

    step = 0
    num_actions = 50
    num_dims = 6

    while True:
        # Generate random trajectories with some continuity
        actions = []
        base = [random.uniform(-1, 1) for _ in range(num_dims)]
        for t in range(num_actions):
            action = [
                base[d] + 0.1 * t + random.gauss(0, 0.05)
                for d in range(num_dims)
            ]
            actions.append(action)

        # Create mock event
        from .metrics import EvActionChunk

        event = EvActionChunk(
            src_control_step=step,
            actions=actions,
            frozen_len=random.randint(5, 15),
            timestamp=time.time(),
        )
        server.on_chunk(event)

        step += num_actions
        time.sleep(interval)


def main():
    """Run the trajectory visualization server standalone."""
    parser = argparse.ArgumentParser(description="RTC Trajectory Visualization Server")
    parser.add_argument("--http_port", type=int, default=8088, help="HTTP server port")
    parser.add_argument("--ws_port", type=int, default=8089, help="WebSocket server port")
    parser.add_argument("--mock", action="store_true", help="Generate mock data for testing")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    server = TrajectoryVizServer(ws_port=args.ws_port, http_port=args.http_port)

    if args.mock:
        # Start mock data generator in background
        mock_thread = threading.Thread(
            target=generate_mock_chunks, args=(server,), daemon=True
        )
        mock_thread.start()
        logger.info("Mock data generator started")

    logger.info(f"Open http://localhost:{args.http_port} in your browser")
    server.start()


if __name__ == "__main__":
    main()
