import asyncio
import concurrent.futures
import threading
import time
from typing import Any, Dict

import cv2
import nats
import nats.errors
import numpy as np

from lerobot.common.robot_devices.cameras.configs import NatsCameraConfig
from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
)
from lerobot.common.utils.utils import capture_timestamp_utc


class NatsCamera(Camera):
    def __init__(self, config: NatsCameraConfig):
        self.config = config
        self.nc: nats.aio.client.Client | None = None
        self.sub: nats.aio.subscription.Subscription | None = None # For one-off reads if needed
        self.latest_image: np.ndarray | None = None
        self.logs: Dict[str, Any] = {}
        self.thread: threading.Thread | None = None
        self.stop_event: threading.Event | None = None
        self.lock = threading.Lock()
        self.is_connected = False # Represents the conceptual connection state of the camera device
        
        self.rotation_code: int | None = None
        if self.config.rotation == 90:
            self.rotation_code = cv2.ROTATE_90_CLOCKWISE
        elif self.config.rotation == -90:
            self.rotation_code = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif self.config.rotation == 180:
            self.rotation_code = cv2.ROTATE_180

    async def _connect_nats(self):
        """Helper async function to connect to NATS."""
        nats_server_url = f"nats://{self.config.nats_server_ip}:{self.config.nats_server_port}"
        try:
            self.nc = await nats.connect(nats_server_url, timeout=self.config.timeout)
        except Exception as e:
            # TODO (aliberts): use specific NATS exceptions once they are well defined in the client
            # e.g. nats.errors.NoServersError, nats.errors.TimeoutError
            # For now, catching generic Exception for broader compatibility.
            raise ConnectionError(f"Failed to connect to NATS server at {nats_server_url}: {e}") from e

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError("NATS camera is already connected.")

        # For the main connection used by the async loop, it's better to connect within the loop's thread.
        # However, if we need a connection for a direct `read()` call before `async_read` is started,
        # we might establish it here or within `read()`.
        # For now, `connect` will conceptually mark the device as ready,
        # and actual NATS connection will happen in `_async_read_loop` or ad-hoc in `read`.
        self.is_connected = True
        # Optionally, can start the async_read loop here if desired,
        # or require explicit call to async_read() to start it.
        # For now, let's keep it separate.

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image. imdecode returned None.")

        if self.config.color_mode == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.rotation_code is not None:
            image = cv2.rotate(image, self.rotation_code)
        
        # Validate shape
        expected_shape = (self.config.height, self.config.width, self.config.channels)
        if image.shape != expected_shape:
            # TODO (aliberts): consider resizing instead of raising error, or making it configurable
            raise ValueError(
                f"Decoded image shape {image.shape} does not match expected shape {expected_shape}."
            )
        return image

    async def _get_message_sync(self) -> nats.aio.client.Msg:
        """Async helper to get a single message. Manages its own connection."""
        if not self.nc or self.nc.is_closed:
            await self._connect_nats()
            if not self.nc: # Should not happen if _connect_nats doesn't raise
                 raise RobotDeviceNotConnectedError("NATS connection not established for sync read.")

        # Ensure nc is not None before using it.
        if self.nc is None:
            raise RobotDeviceNotConnectedError("NATS client self.nc is None even after connect attempt.")

        sub = await self.nc.subscribe(self.config.subject)
        try:
            msg = await sub.next_msg(timeout=self.config.timeout)
        finally:
            await sub.unsubscribe()
            # Decide on connection persistence for sync reads.
            # For simplicity, closing it each time if this is purely ad-hoc.
            # If `read` is called frequently without `async_read`, this is inefficient.
            if self.nc and not self.nc.is_closed:
                await self.nc.close()
            self.nc = None # Mark as closed
        return msg

    def read(self, temporary_color_mode: str | None = None) -> np.ndarray:
        if not self.is_connected: # Conceptual connected state
            raise RobotDeviceNotConnectedError("Call connect() before reading from NATS camera.")

        timestamp_start = time.monotonic()

        # Prioritize image from async loop if active
        if self.thread is not None and self.thread.is_alive() and self.latest_image is not None:
            with self.lock:
                image = self.latest_image.copy()
                # Logs would be updated by the async loop itself
            # TODO (aliberts): Handle temporary_color_mode if required for images from async loop
            if temporary_color_mode is not None and temporary_color_mode != self.config.color_mode:
                if temporary_color_mode == "rgb" and self.config.color_mode == "bgr":
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif temporary_color_mode == "bgr" and self.config.color_mode == "rgb":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image

        # Fallback to synchronous-style read
        try:
            # Run the async helper in a new event loop
            # Python 3.7+ `asyncio.run` can be used.
            # For older versions or specific loop management, `asyncio.new_event_loop()` and `loop.run_until_complete()`
            msg = asyncio.run(self._get_message_sync())
        except nats.errors.TimeoutError:
            raise TimeoutError(f"Timeout waiting for NATS message on subject '{self.config.subject}'") from None
        except ConnectionError as e: # Catch connection errors from _connect_nats via _get_message_sync
             raise RobotDeviceNotConnectedError(f"NATS connection failed during read: {e}") from e
        except Exception as e: # Catch other NATS client errors
            raise RuntimeError(f"Failed to read from NATS: {e}") from e
        
        image = self._decode_image(msg.data)

        timestamp_end = time.monotonic()
        self.logs["delta_timestamp_s"] = timestamp_end - timestamp_start
        self.logs["timestamp_utc"] = capture_timestamp_utc()
        
        # TODO (aliberts): Handle temporary_color_mode if required for sync read
        # The _decode_image already handles the default color_mode.
        # If temporary_color_mode is different, an additional conversion might be needed here.
        # This is simplified for now.

        return image

    def _async_read_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def listen():
            conn: nats.aio.client.Client | None = None
            sub: nats.aio.subscription.Subscription | None = None
            try:
                nats_server_url = f"nats://{self.config.nats_server_ip}:{self.config.nats_server_port}"
                conn = await nats.connect(nats_server_url, timeout=self.config.timeout)
                sub = await conn.subscribe(self.config.subject)
                
                # Signal that connection within the thread is established (optional)
                # self.nc = conn # Assign to self.nc if this loop manages the primary connection

                while not self.stop_event.is_set():
                    try:
                        msg = await sub.next_msg(timeout=1.0) # Timeout to check stop_event
                        timestamp_start = time.monotonic()
                        
                        image = self._decode_image(msg.data)
                        
                        timestamp_end = time.monotonic()
                        
                        with self.lock:
                            self.latest_image = image
                            self.logs["delta_timestamp_s"] = timestamp_end - timestamp_start
                            self.logs["timestamp_utc"] = capture_timestamp_utc()
                            
                    except nats.errors.TimeoutError:
                        continue # Check stop_event and try again
                    except ValueError as e: # Image decoding error
                        print(f"Error decoding image in async loop: {e}") # Or use proper logging
                        # Potentially skip frame or implement error handling (e.g. retry, stop)
                    except Exception as e:
                        print(f"Error in NATS async listen loop: {e}") # Or use proper logging
                        # Depending on error, may need to break or attempt reconnect
                        # For now, continue to allow stop_event check
                        time.sleep(1) # Avoid busy loop on persistent error
            except ConnectionError as e:
                print(f"NATS connection failed in async_read_loop: {e}")
            except Exception as e:
                print(f"Unhandled exception in _async_read_loop listen(): {e}")
            finally:
                if sub:
                    try:
                        await sub.unsubscribe()
                    except Exception as e:
                        print(f"Error unsubscribing in async_read_loop: {e}")
                if conn and not conn.is_closed:
                    try:
                        await conn.close()
                    except Exception as e:
                        print(f"Error closing NATS connection in async_read_loop: {e}")
        
        try:
            loop.run_until_complete(listen())
        finally:
            loop.close()


    def async_read(self, temporary_color_mode: str | None = None) -> np.ndarray | None:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Call connect() before async_read from NATS camera.")

        if self.thread is None or not self.thread.is_alive():
            self.stop_event = threading.Event()
            self.thread = threading.Thread(target=self._async_read_loop, daemon=True)
            self.thread.start()
            # It might take a moment for the first image to be available.
            # Consider adding a small wait or a mechanism to ensure the loop is ready.
            # For now, it might return None if called immediately after start and image isn't ready.

        with self.lock:
            if self.latest_image is not None:
                image = self.latest_image.copy()
            else:
                # First image not yet received, or an error occurred in the loop.
                # Return None or could wait with a timeout (e.g. using a Condition variable)
                return None 
        
        # TODO (aliberts): Handle temporary_color_mode
        if temporary_color_mode is not None and temporary_color_mode != self.config.color_mode:
            if temporary_color_mode == "rgb" and self.config.color_mode == "bgr":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif temporary_color_mode == "bgr" and self.config.color_mode == "rgb":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def disconnect(self):
        if not self.is_connected:
            # Allow calling disconnect multiple times without error, or raise RobotDeviceNotConnectedError
            # For now, just return if not "conceptually" connected.
            return

        if self.thread is not None and self.thread.is_alive():
            if self.stop_event:
                self.stop_event.set()
            self.thread.join(timeout=5.0) # Wait for thread to finish
            if self.thread.is_alive():
                print("Warning: NATS async_read_loop thread did not terminate in time.") # Or log
            self.thread = None
            self.stop_event = None

        # If self.nc was used for synchronous `read` calls and not managed by the async loop,
        # it should be closed there.
        # If the async_read_loop was responsible for self.nc, it handles its closure.
        # For now, we assume the async_read_loop manages its own connection, and sync `read` manages its own.
        # If there was a global self.nc for some reason, it would be closed here.
        # Example:
        # if self.nc and not self.nc.is_closed:
        #     try:
        #         # This needs an event loop if self.nc is an async NATS connection
        #         async def _close_nc():
        #             await self.nc.close()
        #         asyncio.run(_close_nc())
        #     except Exception as e:
        #         print(f"Error closing main NATS connection: {e}") # Or log
        # self.nc = None

        self.is_connected = False
        self.latest_image = None # Clear last image on disconnect

    def get_image_obs(self) -> np.ndarray:
        # This is the primary method expected by the Camera protocol for LeRobot
        # It can use either `read()` or `async_read()` based on desired behavior.
        # Using `async_read()` is generally preferred if continuous streaming is expected.
        if self.config.mock:
            # Return a mock image if in mock mode
            return np.zeros((self.config.height, self.config.width, self.config.channels), dtype=np.uint8)

        img = self.async_read()
        if img is None:
            # Fallback or error if async_read returns None (e.g. not started, or no image yet)
            # Option 1: Try a synchronous read as a fallback
            # print("Async read returned None, attempting synchronous read...")
            # try:
            #    return self.read()
            # except Exception as e:
            #    raise RuntimeError("Failed to get image via async_read (None) and read (error)") from e
            # Option 2: Raise an error or return a default image if async path is expected to work
            raise RuntimeError("Failed to get image from async_read: No image available. Ensure async_read is working correctly.")
        return img
        
    def get_depth_obs(self) -> np.ndarray | None:
        # NATS camera example does not include depth, returning None
        return None

    def get_intrinsics(self) -> np.ndarray | None:
        # NATS camera example does not include intrinsics, returning None
        return None
        
    def get_info(self) -> Dict[str, Any]:
        with self.lock:
            return self.logs.copy()

    def __del__(self):
        # Attempt to clean up resources if the object is garbage collected
        # Note: __del__ can be unreliable. Explicit disconnect is better.
        if hasattr(self, 'is_connected') and self.is_connected:
            try:
                self.disconnect()
            except Exception as e:
                # Suppress errors during __del__ as the interpreter state might be unpredictable
                print(f"Error during NatsCamera.__del__: {e}") # Or log to a safe place
                pass

```
