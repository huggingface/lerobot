"""The cart's two Feetech buses: which port, which id is which joint.

The one copy. Every servo tool reads it; before 2026-10-15 five scripts each
carried their own, and a port or id fixed in one was wrong in the others.
"""

# by-id paths travel with the USB serial adapter, not the port it is in.
PORTS = {
    1: "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A058116-if00",   # left arm + head
    2: "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68009991-if00",   # right arm (+ omniwheels)
}

# id -> joint name, per bus. The wheels exist only with base: omniwheels.
JOINTS = {
    1: {
        1: "left_arm_shoulder_pan", 2: "left_arm_shoulder_lift", 3: "left_arm_elbow_flex",
        4: "left_arm_wrist_flex", 5: "left_arm_wrist_roll", 6: "left_arm_gripper",
        7: "head_motor_1", 8: "head_motor_2",
    },
    2: {
        1: "right_arm_shoulder_pan", 2: "right_arm_shoulder_lift", 3: "right_arm_elbow_flex",
        4: "right_arm_wrist_flex", 5: "right_arm_wrist_roll", 6: "right_arm_gripper",
        7: "base_left_wheel", 8: "base_back_wheel", 9: "base_right_wheel",
    },
}
WHEELS = {"base_left_wheel", "base_back_wheel", "base_right_wheel"}

# joint name -> (bus, id), for everything that is calibrated (not the wheels).
CALIBRATED = {
    name: (bus, mid)
    for bus, table in JOINTS.items()
    for mid, name in table.items()
    if name not in WHEELS
}
