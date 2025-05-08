import mujoco
from mujoco.viewer import launch

# model = mujoco.MjModel.from_xml_path("MJCF/so-arm101.xml")
FILEPATH = "./MJCF/so-arm101/scene.xml"
# FILEPATH = "/media/tsukasa/DATA/works/mujoco_menagerie/aloha/aloha.xml"
model = mujoco.MjModel.from_xml_path(FILEPATH)
data = mujoco.MjData(model)
launch(model, data)
