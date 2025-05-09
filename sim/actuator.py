import os
import time

import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("actuator.xml")
data = mujoco.MjData(model)

waist_id = model.actuator("waist_act").id

# 初期位置の設定
# 初期値は次のフレームにはクリアされて、おそらく目標値は0になるため、
# 次のフレームからdata.ctrlで制御をしないと0に向かっていく
data.qpos[waist_id] = np.deg2rad(90)

try:
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False) as viewer:
        while viewer.is_running():
            angle = np.deg2rad(90)
            data.ctrl[waist_id] = angle
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)
finally:
    os._exit(0)
