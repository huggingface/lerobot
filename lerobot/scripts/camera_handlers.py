import logging
import os
import queue
import threading
import time
from datetime import datetime
import cv2
import PySpin

class CameraHandlerError(Exception):
    pass


class Camera:
    def __init__(self, output_dir, capture_interval_seconds=None):
        self.output_dir = output_dir
        self.capture_interval_seconds = capture_interval_seconds

        self.latest_image = None  # Buffer to store the latest image
        self.latest_image_timestamp = None

        self.lock = threading.Lock()  # Lock to synchronize access to the buffer

        self.capture_thread = None
        self.running = False
        self.stop_event = threading.Event()  # Event to signal the threads to stop

        # Queue for storing images to be saved
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(
            target=self._save_images_from_queue, daemon=True
        )
        self.save_thread.start()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.open()
        self.start_capture_thread()

    def get_timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def open(self):
        raise NotImplementedError("Subclasses should implement this method")

    def capture_single_frame(self):
        raise NotImplementedError("Subclasses should implement this method")

    def release(self):
        raise NotImplementedError("Subclasses should implement this method")

    def save_latest_frame(self):
        with self.lock:
            if self.latest_image is not None:
                timestamp = self.latest_image_timestamp
                frame_filename = os.path.join(self.output_dir, f"{timestamp}.png")
                self.save_queue.put((self.latest_image.copy(), frame_filename))
                return frame_filename, timestamp, self.latest_image.copy()
            else:
                return None, None, None

    def _save_images_from_queue(self):
        while not self.stop_event.is_set() or not self.save_queue.empty():
            try:
                image, filename = self.save_queue.get(
                    timeout=1
                )  # Timeout for responsiveness
                cv2.imwrite(filename, image)
                self.save_queue.task_done()
            except queue.Empty:
                continue

    def start_capture_thread(self):
        if not self.running:
            self.running = True
            self.capture_thread = threading.Thread(
                target=self.capture_frames, daemon=True
            )
            self.capture_thread.start()
            self.logger.info("Capture thread started.")

    def capture_frames(self):
        try:
            while not self.stop_event.is_set():
                self.capture_single_frame()
                if self.capture_interval_seconds is not None:
                    time.sleep(self.capture_interval_seconds)
        except Exception as e:
            self.logger.error(f"Error in capture_frames: {e}")
        finally:
            self.logger.info("Capture thread stopped.")

    def stop_capture_thread(self):
        if self.running:
            self.running = False
            self.stop_event.set()
            if self.capture_thread is not None:
                self.capture_thread.join()
                self.logger.info("Capture thread terminated.")
        else:
            self.logger.info(
                "Received stop_capture_thread. Capture thread not running."
            )

    def stop_save_thread(self):
        self.stop_event.set()
        self.save_queue.join()  #
        self.save_thread.join()
        self.logger.info("Save thread terminated.")

    def __del__(self):
        self.stop_capture_thread()
        self.stop_save_thread()
        self.release()
        self.logger.info("Camera object destroyed.")


class USBCamera(Camera):
    def __init__(self, device_path, output_dir, capture_interval_seconds=None):
        self.device_path = device_path
        self.capture_stream = None
        super().__init__(output_dir, capture_interval_seconds)

    def open(self):
        self.capture_stream = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        if not self.capture_stream.isOpened():
            raise CameraHandlerError(f"Could not open video device {self.device_path}")
        exposure = self.capture_stream.get(cv2.CAP_PROP_EXPOSURE)
        gain = self.capture_stream.get(cv2.CAP_PROP_GAIN)
        brightness = self.capture_stream.get(cv2.CAP_PROP_BRIGHTNESS)
        contrast = self.capture_stream.get(cv2.CAP_PROP_CONTRAST)
        saturation = self.capture_stream.get(cv2.CAP_PROP_SATURATION)
        self.logger.info(
            f"Exposure: {exposure}, Gain: {gain}, Brightness: {brightness}, "
            f"Contrast: {contrast}, Saturation: {saturation}"
        )

    def capture_single_frame(self):
        ret, frame = self.capture_stream.read()
        if not ret:
            self.logger.warning(f"Error: Could not read frame from {self.device_path}")
            return None

        with self.lock:
            self.latest_image = frame
            self.latest_image_timestamp = self.get_timestamp()

    def release(self):
        self.stop_capture_thread()
        self.stop_save_thread()
        if self.capture_stream:
            self.capture_stream.release()
            self.capture_stream = None
        self.logger.info(f"Released {self.device_path}")


class SpinnakerCamera(Camera):
    def __init__(self, output_dir, capture_interval_seconds=None, serial_number=None):
        self.system = None
        self.cam_list = None
        self.cam = None
        self.serial_number = serial_number
        super().__init__(output_dir, capture_interval_seconds)

    def open(self):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()

        if self.cam_list.GetSize() == 0:
            raise CameraHandlerError("No cameras found using Spinnaker SDK.")

        if self.serial_number:
            # Look for the camera with the specified serial number
            for i in range(self.cam_list.GetSize()):
                cam = self.cam_list.GetByIndex(i)
                if cam.GetUniqueID() == self.serial_number:
                    self.cam = cam
                    break
            if self.cam is None:
                raise CameraHandlerError(
                    f"Camera with serial number {self.serial_number} not found."
                )
        elif self.cam_list.GetSize() > 1:
            raise CameraHandlerError(
                f"{self.cam_list.GetSize()} cameras found but no serial number provided."
            )
        else:
            self.logger.info(
                "No serial number provided. Using the first available camera."
            )
            self.cam = self.cam_list[0]

        self.cam.Init()
        self.cam.BeginAcquisition()

    def capture_single_frame(self):
        try:
            image_result = self.cam.GetNextImage()
            if image_result.IsIncomplete():
                self.logger.warning(
                    f"Image incomplete with image status {image_result.GetImageStatus()}"
                )
                return None

            image_data = image_result.GetNDArray()
            with self.lock:
                self.latest_image = image_data
                self.latest_image_timestamp = self.get_timestamp()

        except PySpin.SpinnakerException as ex:
            self.logger.error(f"Error: {ex}")
            return None

    def release(self):
        self.stop_capture_thread()
        self.stop_save_thread()
        if self.cam:
            self.logger.info(
                f"Stopping acquisition for camera with serial number {self.serial_number}."
            )
            self.cam.EndAcquisition()
            self.cam.DeInit()
            del self.cam
            self.cam = None
        if self.cam_list:
            self.logger.info("Clearing camera list.")
            self.cam_list.Clear()
            self.cam_list = None
        if self.system:
            self.logger.info("Releasing system resources.")
            self.system.ReleaseInstance()
            self.system = None
        if self.serial_number:
            self.logger.info(
                "Released Spinnaker camera with serial number " f"{self.serial_number}."
            )
        else:
            self.logger.info("Released Spinnaker camera.")


