import threading
from threading import Thread

import numpy as np
import cv2
import pyorbbecsdk as OB
from typing import Union, Any, Optional
import time

from lerobot.common.robot_devices.cameras.configs import OrbbecCameraConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
)
from lerobot.common.utils.utils import capture_timestamp_utc
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
    busy_wait,
)
def frame_to_bgr_image(frame: OB.VideoFrame) -> Union[Optional[np.array], Any]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OB.OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OB.OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OB.OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OB.OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OB.OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OB.OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
        return image
    elif color_format == OB.OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
        return image
    elif color_format == OB.OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image



class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result

SERIAL_NUMBER_INDEX = 1

MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

class OrbbecCamera:
    def __init__(
        self,
        config: OrbbecCameraConfig,
    ):
        self.config = config
        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.mock = config.mock
        self.index = None
        self.channels = 3
        self.Hi_resolution_mode = config.Hi_resolution_mode
        
        self.depth_height = None 
        self.color_height = None
        if self.use_depth:
            match self.width:
                case 640:
                    self.depth_height = 400
                    self.color_height = 480
                case 1280:
                    self.depth_height = 800
                    self.color_height = 720

        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.depth_map = None
        self.logs = {}
        self.temporal_filter = TemporalFilter(config.TemporalFilter_alpha)
        if self.mock:
            import tests.mock_cv2 as cv2
        else:
            import cv2



    def fuse_color_and_depth(self,color_image, depth_rgb_packed) -> np.ndarray:
        if color_image.shape[1] != depth_rgb_packed.shape[1]:
            raise ValueError("Width mismatch between color and depth images.")

        stacked = np.vstack((color_image, depth_rgb_packed))  
        return stacked
    def load_depth_config(self):
        try:
            profile_list = self.camera.get_stream_profile_list(OB.OBSensorType.DEPTH_SENSOR)
            assert profile_list is not None
            depth_profile = profile_list.get_video_stream_profile(self.width, self.depth_height, OB.OBFormat.Y16, self.fps)
            assert depth_profile is not None
            print("\033[32mDEPTH Profile Loaded:\033[0m", depth_profile)
            self.OBconfig.enable_stream(depth_profile)
        except Exception as e:
            print(e)
            return
        
    def load_color_config(self):
        try:
            profile_list = self.camera.get_stream_profile_list(OB.OBSensorType.COLOR_SENSOR)
            assert profile_list is not None
            color_profile = profile_list.get_video_stream_profile(self.width, self.color_height, OB.OBFormat.RGB, self.fps)
            assert color_profile is not None
            print("\033[32mCOLOR Profile Loaded:\033[0m ", color_profile)
            self.OBconfig.enable_stream(color_profile)
        except Exception as e:
            print(e)
            return
        
    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
               "OrbbecCamera is readyConnected"
            )
        if self.mock:
            print("Waring!!MockMode is under repairing")
            return
            
        print("\033[32mHello! Orbbec!\033[0m")

        self.OBconfig = OB.Config()
        self.camera = OB.Pipeline()

        if self.use_depth:
            self.load_depth_config()

        self.load_color_config()
        
        self.camera.start(self.OBconfig)

        self.is_connected = True
        print("\033[32mCAMERA CONNECTED\033[0m ")
        time.sleep(5)

    def HandleDepth(self, depth_frame):
        if depth_frame is None:
            print("No depth frame received")
            return None

        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()

        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((height, width))

        # Apply temporal filter
        filtered_depth_data = self.temporal_filter.process(depth_data)
        # Convert to float32 and apply scale
        #filtered_depth_data = filtered_depth_data.astype(np.float32) * scale
        filtered_depth_data = filtered_depth_data.astype(np.uint16)
        
        # depth_mm = (filtered_depth_data * 1000).astype(np.uint32) 
        if self.Hi_resolution_mode:
            R = ((filtered_depth_data >> 8) & 0xFF).astype(np.uint8) 
            G = (filtered_depth_data & 0xFF).astype(np.uint8)        
            B = np.zeros_like(R, dtype=np.uint8)              

            filtered_depth_data = cv2.merge([B, G, R])  

        else:
            filtered_depth_data = cv2.normalize(filtered_depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            filtered_depth_data = cv2.applyColorMap(filtered_depth_data, cv2.COLORMAP_JET)
        return filtered_depth_data

    def read(self):
            start_time = time.perf_counter()
            frames = self.camera.wait_for_frames(1000)
            if frames is None:
                print("No frames received")
            if self.use_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame is None:
                    print("No depth frame received")
                    
                self.depth_map = self.HandleDepth(depth_frame)
            color_frame = frames.get_color_frame()
         
            if color_frame is None:
                print("No color frame received")
                
            self.color_image = frame_to_bgr_image(color_frame)
            
            if self.color_image is None:
                print("failed to convert frame to image")

            self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

            # log the utc time at which the image was received
            self.logs["timestamp_utc"] = capture_timestamp_utc()

        
    def read_loop(self):
        print("开始尝试")
        while not self.stop_event.is_set():
            try:
                self.read()
            except Exception as e:
                print(e)
                break  

    def async_read(self):
        if self.thread is None or not self.thread.is_alive():
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.read_loop, args=(),daemon=True)
            self.thread.start()

        num_tries = 0
        while self.color_image is None:
           
            # TODO(rcadene, aliberts): intelrealsense has diverged compared to opencv over here
            num_tries += 1
            time.sleep(1 / self.fps)
            #if num_tries > self.fps and (self.thread.ident is None or not self.thread.is_alive()):
                #raise Exception(
                #print(   "The thread responsible for `self.async_read()` took too much time to start. There might be an issue. Verify that `self.thread.start()` has been called.")######可能一直报错
        
        if self.use_depth:
            return self.fuse_color_and_depth(self.color_image, self.depth_map)

        else:
            return self.color_image
    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"IntelRealSenseCamera({self.serial_number}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is not None and self.thread.is_alive():
            # wait for the thread to finish
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

        self.camera.stop()
        self.camera = None

        self.is_connected = False
    def test_read(self):
        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.test_loop, args=())
            self.thread.daemon = True
            self.thread.start()
if __name__ == "__main__":
    
    # Create a configuration for the OrbbecCamera
    config = OrbbecCameraConfig(
        fps=30,
        width=640,
        height=480,
        color_mode="bgr",
        use_depth=True,
        mock=False,
        index=0,
    )

    # Initialize the camera
    camera = OrbbecCamera(config)
    # Connect to the camera
    camera.connect()
    time.sleep(10)

    # Start asynchronous reading
    while True:
        camera.async_read()
        

   
