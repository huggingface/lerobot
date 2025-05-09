import os
import time
import traceback

import mujoco
import mujoco.viewer
import numpy as np

FILEPATH = "./MJCF/so-arm101/scene.xml"
model = mujoco.MjModel.from_xml_path(FILEPATH)
data = mujoco.MjData(model)

print("---------- joints ----------")
joints = [model.joint(i) for i in range(model.njnt)]
for joint in joints:
    print(joint.id, joint.name, joint.pos)

is_running = True


def key_callback(keycode):
    if chr(keycode) == "Q":
        global is_running
        is_running = False
        print("Finish!")


# 目標位置の表示
# goal_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal_site")
goal_site_id = model.site("goal_site").id
assert goal_site_id >= 0
model.site_pos[goal_site_id] = np.array([0.5, 0.5, 0.5])

# エンドエフェクタの位置の表示用
ee_site_id = model.site("ee_site").id
assert ee_site_id >= 0

# 位置を取得したいsiteのid
pinch_site_id = model.site("pinch").id
assert pinch_site_id >= 0


amplitude = np.deg2rad(20)  # 振幅 ±30度
frequency = 0.4

try:
    viewer = mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        # show_right_ui=False,
        key_callback=key_callback,
    )
    viewer.sync()

    waist_id = model.actuator("waist").id
    shoulder_id = model.actuator("shoulder").id

    while is_running and viewer.is_running():
        target_angle = amplitude * np.sin(2 * np.pi * frequency * data.time)
        # target_angle = np.deg2rad(90)
        data.ctrl[waist_id] = target_angle
        data.ctrl[shoulder_id] = target_angle
        # これなら動く
        # data.qpos[waist_id] = target_angle
        # mujoco.mj_forward(model, data)
        # print(target_angle)

        # エンドエフェクタの位置を取得しボールの位置を変える
        pinch_pos = data.site_xpos[pinch_site_id]
        model.site_pos[ee_site_id] = pinch_pos

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
    viewer.close()
except Exception as e:
    print(e, traceback.print_exc())
finally:
    # 終了時にスレッドが残ってしまいプロセスが終わらないので強制終了
    os._exit(0)
