import mujoco

class SimCamera:

    def __init__(self, model, data, id_camera, camera_index, fps=30, width=640, height=480):
        self.model = model
        self.data  = data
        self.camera_index = camera_index
        self.id_camera = id_camera
        self.is_connected = False
        self.fps = fps
        self.width = width
        self.height = height

        self.logs = {}
        self.logs["delta_timestamp_s"] = 1.0 / self.fps
        
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

    def async_read(self):
            self.renderer.update_scene(self.data, camera=self.id_camera)
            return self.renderer.render()
