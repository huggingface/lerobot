#!/usr/bin/env python3
# 注意demo无法直接运行，需要pip安装sdk后才能运行
# 设置机械臂重置，需要在mit或者示教模式切换为位置速度控制模式时执行
import time
from piper_sdk import C_PiperInterface_V2

# 测试代码
if __name__ == "__main__":
    piper_1 = C_PiperInterface_V2("can_master", judge_flag=True)  # type: ignore
    piper_2 = C_PiperInterface_V2("can_follower", judge_flag=True)  # type: ignore
    piper_1.ConnectPort()
    piper_2.ConnectPort()

    while(piper_1.DisablePiper()):
        time.sleep(0.01)
    print("piper_1 失能成功!!!!")

    while(piper_2.DisablePiper()):
        time.sleep(0.01)
    print("piper_2 失能成功!!!!")
    piper_1.MotionCtrl_1(0x02,0,0)#恢复
    piper_2.MotionCtrl_1(0x02,0,0)#恢复
    time.sleep(0.5)
