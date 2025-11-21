import multiprocessing as mp
from multiprocessing import shared_memory
import time
from typing import Any, Dict

import numpy as np

from .sensor_utils import ImageMessageSchema, ImageUtils, SensorServer


def get_multiprocessing_info(verbose: bool = True):
    """Get information about multiprocessing start methods"""

    if verbose:
        print(f"Available start methods: {mp.get_all_start_methods()}")
    return mp.get_start_method()


class ImagePublishProcess:
    """Subprocess for publishing images using shared memory and ZMQ"""

    def __init__(
        self,
        camera_configs: Dict[str, Any],
        image_dt: float,
        zmq_port: int = 5555,
        start_method: str = "spawn",
        verbose: bool = False,
    ):
        self.camera_configs = camera_configs
        self.image_dt = image_dt
        self.zmq_port = zmq_port
        self.verbose = verbose
        self.shared_memory_blocks = {}
        self.shared_memory_info = {}
        self.process = None

        # Use specific context to avoid global state pollution
        self.mp_context = mp.get_context(start_method)
        if self.verbose:
            print(f"Using multiprocessing context: {start_method}")

        self.stop_event = self.mp_context.Event()
        self.data_ready_event = self.mp_context.Event()

        # Ensure events start in correct state
        self.stop_event.clear()
        self.data_ready_event.clear()

        if self.verbose:
            print(f"Initial stop_event state: {self.stop_event.is_set()}")
            print(f"Initial data_ready_event state: {self.data_ready_event.is_set()}")

        # Calculate shared memory requirements for each camera
        for camera_name, camera_config in camera_configs.items():
            height = camera_config["height"]
            width = camera_config["width"]
            # RGB image: height * width * 3 (uint8)
            size = height * width * 3

            # Create shared memory block
            shm = shared_memory.SharedMemory(create=True, size=size)
            self.shared_memory_blocks[camera_name] = shm
            self.shared_memory_info[camera_name] = {
                "name": shm.name,
                "size": size,
                "shape": (height, width, 3),
                "dtype": np.uint8,
            }

    def start_process(self):
        """Start the image publishing subprocess"""
        if self.verbose:
            print(f"Starting subprocess with stop_event state: {self.stop_event.is_set()}")
        self.process = self.mp_context.Process(
            target=self._image_publish_worker,
            args=(
                self.shared_memory_info,
                self.image_dt,
                self.zmq_port,
                self.stop_event,
                self.data_ready_event,
                self.verbose,
            ),
        )
        self.process.start()
        if self.verbose:
            print(f"Subprocess started, PID: {self.process.pid}")

    def update_shared_memory(self, render_caches: Dict[str, np.ndarray]):
        """Update shared memory with new rendered images"""
        images_updated = 0
        for camera_name in self.camera_configs.keys():
            image_key = f"{camera_name}_image"
            if image_key in render_caches:
                image = render_caches[image_key]

                # Ensure image is uint8 and has correct shape
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)

                # Get shared memory array
                shm = self.shared_memory_blocks[camera_name]
                shared_array = np.ndarray(
                    self.shared_memory_info[camera_name]["shape"],
                    dtype=self.shared_memory_info[camera_name]["dtype"],
                    buffer=shm.buf,
                )

                # Copy image data to shared memory atomically
                np.copyto(shared_array, image)
                images_updated += 1

        # Signal that new data is ready only after all images are written
        if images_updated > 0:
            if self.verbose:
                print(f"Main process: Updated {images_updated} images, setting data_ready_event")
            self.data_ready_event.set()
        elif self.verbose:
            print(
                "Main process: No images to update. "
                "please check if camera configs are provided and the renderer is properly initialized"
            )

    def stop(self):
        """Stop the image publishing subprocess"""
        if self.verbose:
            print("Stopping image publishing subprocess...")
        self.stop_event.set()

        if self.process and self.process.is_alive():
            # Give the process time to clean up gracefully
            self.process.join(timeout=5)
            if self.process.is_alive():
                if self.verbose:
                    print("Subprocess didn't stop gracefully, terminating...")
                self.process.terminate()
                self.process.join(timeout=2)
                if self.process.is_alive():
                    if self.verbose:
                        print("Force killing subprocess...")
                    self.process.kill()
                    self.process.join()

        # Clean up shared memory
        for camera_name, shm in self.shared_memory_blocks.items():
            try:
                shm.close()
                shm.unlink()
                if self.verbose:
                    print(f"Cleaned up shared memory for {camera_name}")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to cleanup shared memory for {camera_name}: {e}")

        self.shared_memory_blocks.clear()
        if self.verbose:
            print("Image publishing subprocess stopped and cleaned up")

    @staticmethod
    def _image_publish_worker(
        shared_memory_info, image_dt, zmq_port, stop_event, data_ready_event, verbose
    ):
        """Worker function that runs in the subprocess"""
        # Import dependencies within worker (needed for multiprocessing spawn mode)
        from .sensor_utils import ImageMessageSchema, ImageUtils, SensorServer
        
        if verbose:
            print(f"Worker started! PID: {__import__('os').getpid()}")
            print(f"Worker stop_event state at start: {stop_event.is_set()}")
            print(f"Worker data_ready_event state at start: {data_ready_event.is_set()}")

        try:
            # Initialize ZMQ sensor server
            sensor_server = SensorServer()
            sensor_server.start_server(port=zmq_port)

            # Connect to shared memory blocks
            shared_arrays = {}
            shm_blocks = {}
            for camera_name, info in shared_memory_info.items():
                shm = shared_memory.SharedMemory(name=info["name"])
                shm_blocks[camera_name] = shm
                shared_arrays[camera_name] = np.ndarray(
                    info["shape"], dtype=info["dtype"], buffer=shm.buf
                )

            print(
                f"Image publishing subprocess started with {len(shared_arrays)} cameras on ZMQ port {zmq_port}"
            )

            loop_count = 0
            last_data_time = time.time()

            while not stop_event.is_set():
                loop_count += 1

                # Wait for new data with shorter timeout for better responsiveness
                timeout = min(image_dt, 0.1)  # Max 100ms timeout
                data_available = data_ready_event.wait(timeout=timeout)

                current_time = time.time()

                if data_available:
                    data_ready_event.clear()
                    if loop_count % 50 == 0:
                        print("Image publish frequency: ", 1 / (current_time - last_data_time))
                    last_data_time = current_time

                    # Collect all camera images and serialize them
                    try:
                        # Copy all images atomically at once
                        image_copies = {name: arr.copy() for name, arr in shared_arrays.items()}

                        # Create message with all camera images
                        message_dict = {
                            "images": image_copies,
                            "timestamps": {name: current_time for name in image_copies.keys()},
                        }

                        # Create ImageMessageSchema and serialize
                        image_msg = ImageMessageSchema(
                            timestamps=message_dict.get("timestamps"),
                            images=message_dict.get("images", None),
                        )

                        # Serialize and send via ZMQ
                        serialized_data = image_msg.serialize()

                        # Add individual camera images to the message
                        for camera_name, image_copy in image_copies.items():
                            serialized_data[f"{camera_name}"] = ImageUtils.encode_image(image_copy)

                        sensor_server.send_message(serialized_data)

                    except Exception as e:
                        print(f"Error publishing images: {e}")

                elif verbose and loop_count % 10 == 0:
                    print(f"Subprocess: Still waiting for data... (iteration {loop_count})")

                # Small sleep to prevent busy waiting when no data
                if not data_available:
                    time.sleep(0.001)

        except KeyboardInterrupt:
            print("Image publisher interrupted by user")
        finally:
            # Clean up
            try:
                for shm in shm_blocks.values():
                    shm.close()
                sensor_server.stop_server()
            except Exception as e:
                print(f"Error during subprocess cleanup: {e}")
            if verbose:
                print("Image publish subprocess stopped")
