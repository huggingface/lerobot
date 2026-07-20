from joyconrobotics import JoyconRobotics
import time

# 继承JoyconRobotics类来修改控制逻辑
class FixedAxesJoyconRobotics(JoyconRobotics):
    def __init__(self, 
                 device="right", 
                 joycon_stick_v_0=2300, 
                 joycon_stick_h_0=2000, 
                 **kwargs):
        # 调用父类构造函数
        super().__init__(device, **kwargs)
        
        # 存储摇杆中心值
        self.joycon_stick_v_0 = joycon_stick_v_0
        self.joycon_stick_h_0 = joycon_stick_h_0
    
    def common_update(self):
        # 修改后的更新逻辑：摇杆只控制固定轴向
        
        # 垂直摇杆：只控制X轴（前后）
        joycon_stick_v = self.joycon.get_stick_right_vertical() if self.joycon.is_right() else self.joycon.get_stick_left_vertical()
        joycon_stick_v_threshold = 300
        joycon_stick_v_range = 1000
        # print(f"joycon_stick_v: {joycon_stick_v}")
        if joycon_stick_v > joycon_stick_v_threshold + self.joycon_stick_v_0:
            self.position[0] += 0.001 * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[0] 
        elif joycon_stick_v < self.joycon_stick_v_0 - joycon_stick_v_threshold:
            self.position[0] += 0.001 * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[0] 
        
        # 水平摇杆：只控制Y轴（左右）  
        joycon_stick_h = self.joycon.get_stick_right_horizontal() if self.joycon.is_right() else self.joycon.get_stick_left_horizontal()
        joycon_stick_h_threshold = 300
        joycon_stick_h_range = 1000
        # print(f"stick_h: {joycon_stick_h}, stick_v: {joycon_stick_v}")
        if joycon_stick_h > joycon_stick_h_threshold + self.joycon_stick_h_0:
            self.position[1] += 0.001 * (joycon_stick_h - self.joycon_stick_h_0) / joycon_stick_h_range * self.dof_speed[1] * self.direction_reverse[1]
        elif joycon_stick_h < self.joycon_stick_h_0 - joycon_stick_h_threshold:
            self.position[1] += 0.001 * (joycon_stick_h - self.joycon_stick_h_0) / joycon_stick_h_range * self.dof_speed[1] * self.direction_reverse[1]
        
        # Z轴只通过按钮控制
        joycon_button_up = self.joycon.get_button_r() if self.joycon.is_right() else self.joycon.get_button_l()
        if joycon_button_up == 1:
            self.position[2] += 0.001 * self.dof_speed[2] 
        
        joycon_button_down = self.joycon.get_button_r_stick() if self.joycon.is_right() else self.joycon.get_button_l_stick()
        if joycon_button_down == 1:
            self.position[2] -= 0.001 * self.dof_speed[2] 

        # 其他按钮控制（复制原来的逻辑）
        joycon_button_xup = self.joycon.get_button_x() if self.joycon.is_right() else self.joycon.get_button_up()
        joycon_button_xback = self.joycon.get_button_b() if self.joycon.is_right() else self.joycon.get_button_down()
        if joycon_button_xup == 1:
            self.position[0] += 0.001 * self.dof_speed[0]
        elif joycon_button_xback == 1:
            self.position[0] -= 0.001 * self.dof_speed[0]
        
        # Home按钮重置逻辑（简化版）
        joycon_button_home = self.joycon.get_button_home() if self.joycon.is_right() else self.joycon.get_button_capture()
        if joycon_button_home == 1:
            self.position = self.offset_position_m.copy()
        
        # 夹爪控制逻辑（复制原来的）
        for event_type, status in self.button.events():
            if (self.joycon.is_right() and event_type == 'plus' and status == 1) or (self.joycon.is_left() and event_type == 'minus' and status == 1):
                self.reset_button = 1
                self.reset_joycon()
            elif self.joycon.is_right() and event_type == 'a':
                self.next_episode_button = status
            elif self.joycon.is_right() and event_type == 'y':
                self.restart_episode_button = status
            elif ((self.joycon.is_right() and event_type == 'zr') or (self.joycon.is_left() and event_type == 'zl')) and not self.change_down_to_gripper:
                self.gripper_toggle_button = status
            elif ((self.joycon.is_right() and event_type == 'stick_r_btn') or (self.joycon.is_left() and event_type == 'stick_l_btn')) and self.change_down_to_gripper:
                self.gripper_toggle_button = status
            else: 
                self.reset_button = 0
            
        if self.gripper_toggle_button == 1 :
            if self.gripper_state == self.gripper_open:
                self.gripper_state = self.gripper_close
            else:
                self.gripper_state = self.gripper_open
            self.gripper_toggle_button = 0

        # 按钮控制状态
        if self.joycon.is_right():
            if self.next_episode_button == 1:
                self.button_control = 1
            elif self.restart_episode_button == 1:
                self.button_control = -1
            elif self.reset_button == 1:
                self.button_control = 8
            else:
                self.button_control = 0
        
        return self.position, self.gripper_state, self.button_control

# 使用修改后的控制类
joyconrobotics_left = FixedAxesJoyconRobotics(
    device="left",  # 改为左手控制器
    joycon_stick_v_0=2300,  # 垂直摇杆中心值
    joycon_stick_h_0=2000,  # 水平摇杆中心值
    dof_speed=[2, 2, 2, 1, 1, 1]
)
# joyconrobotics_right = FixedAxesJoyconRobotics(
#     device="right",
#     joycon_stick_v_0=1900,  # 垂直摇杆中心值
#     joycon_stick_h_0=2100,  # 水平摇杆中心值
#     dof_speed=[2, 2, 2, 1, 1, 1]
# )

print("固定轴向控制测试:")
print("垂直摇杆: 只控制X轴（前后）")  
print("水平摇杆: 只控制Y轴（左右）")
print("L按钮: Z轴上升")  # 改为L按钮
print("摇杆按钮: Z轴下降")
print("Capture按钮: 重置位置")  # 改为Capture按钮
print("ZL按钮: 切换夹爪")  # 改为ZL按钮
print("按Ctrl+C停止")
print()

for i in range(10000):
    pose_left, gripper_left, control_button_left = joyconrobotics_left.get_control()  # 改变量名
    # pose_right, gripper_right, control_button_right = joyconrobotics_right.get_control()  # 改变量名
    x_left, y_left, z_left, roll_left, pitch_left, yaw_left = pose_left
    # x_right, y_right, z_right, roll_right, pitch_right, yaw_right = pose_right
    print(f'pos_left={x_left:.3f}, {y_left:.3f}, {z_left:.3f}, Rot_left={roll_left:.3f}, {pitch_left:.3f}, {yaw_left:.3f}, gripper_left={gripper_left}, control_button_left={control_button_left}')
    # print(f'pos_right={x_right:.3f}, {y_right:.3f}, {z_right:.3f}, Rot_right={roll_right:.3f}, {pitch_right:.3f}, {yaw_right:.3f}, gripper_right={gripper_right}, control_button_right={control_button_right}')
    time.sleep(0.02)

joyconrobotics_left.disconnect()  # 改变量名 
