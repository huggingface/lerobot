import os
import time
import traceback

import mujoco
import mujoco.viewer
import numpy as np

FILEPATH = "./MJCF/so-arm101/scene.xml"
model = mujoco.MjModel.from_xml_path(FILEPATH)
data = mujoco.MjData(model)

# print("==================================")
# print("for model")
# print("==================================")
# print("---------- geoms ----------")
# geoms = [model.geom(i) for i in range(model.ngeom)]
# for geom in geoms:
#     print(geom.name)

print("---------- joints ----------")
joints = [model.joint(i) for i in range(model.njnt)]
for joint in joints:
    print(joint.name, joint.pos)

# print("==================================")
# print("for data")
# print("==================================")
# for _ in range(1000):
#     mujoco.mj_step(model, data)
#     # print(data.qpos)
#     # print(data.qvel)


is_running = True


def key_callback(keycode):
    if chr(keycode) == "Q":
        global is_running
        is_running = False
        print("Finish!")


# 目標位置の表示
goal_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal_site")
assert goal_site_id >= 0
model.site_pos[goal_site_id] = np.array([0.5, 0.5, 0.5])

# エンドエフェクタの位置の表示
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
# model.site_pos[ee_site_id] = np.array([0.3, 0.3, 0.3])
assert ee_site_id >= 0

# 位置を取得したいsiteのid
pinch_site_id = model.site("pinch").id
assert pinch_site_id >= 0

try:
    viewer = mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    )
    viewer.sync()
    while is_running and viewer.is_running():
        mujoco.mj_step(model, data)
        pinch_pos = data.site_xpos[pinch_site_id]
        model.site_pos[ee_site_id] = pinch_pos
        viewer.sync()
        time.sleep(model.opt.timestep)
    viewer.close()
except Exception as e:
    print(e, traceback.print_exc())
finally:
    os._exit(0)


# goal_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal_site")
# model.site_pos[goal_site_id] = np.array([0.1, 0.1, 0.1])


# pinch_id = mujoco.mj_name2id(
#     model,
#     mujoco.mjtObj.mjOBJ_SITE,
#     "pinch",
# )
# pinch_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
# model.site_pos[pinch_site_id] = model.site_pos[pinch_id]

# # print()
# for j in data.joint():
#     print(i)
