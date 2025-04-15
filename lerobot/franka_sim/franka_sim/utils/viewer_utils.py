import mujoco.viewer

class DualMujocoViewer:
    def __init__(
            self,
            model,
            data,
            key_callback=None
    ):
        self.model = model
        self.data = data
        self.viewer_1 = None
        self.viewer_2 = None
        self.key_callback = key_callback

    def __enter__(self):
        self.launch()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def launch(self):
        self.viewer_1 = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=self.key_callback
        )
        self.viewer_2 = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=self.key_callback
        )

    def is_running(self):
        return self.viewer_1.is_running() and self.viewer_2.is_running()

    def sync(self):
        if self.viewer_1:
            self.viewer_1.sync()
        if self.viewer_2:
            self.viewer_2.sync()

    def close(self):
        import glfw
        if self.viewer_1:
            self.viewer_1.close()
        if self.viewer_2:
            self.viewer_2.close()
        glfw.terminate()
