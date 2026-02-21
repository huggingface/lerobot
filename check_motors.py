from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.configs.robot import MotorConfig

# Define at least one motor properly
motors = {
    "one": MotorConfig(id=1, model="sts3215"),
    "two": MotorConfig(id=1, model="sts3215"),
    "three": MotorConfig(id=1, model="sts3215"),
    "four": MotorConfig(id=1, model="sts3215"),
    "five": MotorConfig(id=1, model="sts3215"),
    "six": MotorConfig(id=1, model="sts3215"),
}

try:
    bus = FeetechMotorsBus(port="/dev/tty.usbmodem5AE60840061", motors=motors)
    bus.connect()
    print("✅ SUCCESS: Connection established and Motor ID 1 found!")
except Exception as e:
    print(f"❌ FAILED: {e}")
