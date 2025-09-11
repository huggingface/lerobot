import time

from lerobot.robots.viperx import ViperX, ViperXConfig
from lerobot.teleoperators.widowx import WidowX, WidowXConfig

config_follower_right = ViperXConfig(
    port="/dev/tty.usbserial-FT891KBG",
    id="viperx_right",
    max_relative_target=10.0,  # increased from default 5.0 to 10.0
)

config_leader_right = WidowXConfig(
    port="/dev/tty.usbserial-FT89FM77",
    id="widowx_right",
    gripper_motor="xc430-w150",
)

config_follower_left = ViperXConfig(
    port="/dev/tty.usbserial-FT89FM09",
    id="viperx_left",
    max_relative_target=10.0,  # increased from default 5.0 to 10.0
)

config_leader_left = WidowXConfig(
    port="/dev/tty.usbserial-FT891KPN",
    id="widowx_left",
    gripper_motor="xl430-w250",
)

follower_right = ViperX(config_follower_right)
follower_right.connect()

leader_right = WidowX(config_leader_right)
leader_right.connect()

follower_left = ViperX(config_follower_left)
follower_left.connect()

leader_left = WidowX(config_leader_left)
leader_left.connect()

while True:
    act_right = leader_right.get_action()
    obs_right = follower_right.get_observation()

    act_left = leader_left.get_action()
    obs_left = follower_left.get_observation()

    print("=" * 60)
    print("ACTION (Leader Right):")
    for key, value in act_right.items():
        print(f"  {key:20}: {value:8.3f}")

    print("\nOBSERVATION (Follower Right):")
    for key, value in obs_right.items():
        print(f"  {key:20}: {value:8.3f}")
    print("=" * 60)
    print("ACTION (Leader Left):")
    for key, value in act_left.items():
        print(f"  {key:20}: {value:8.3f}")

    print("\nOBSERVATION (Follower Left):")
    for key, value in obs_left.items():
        print(f"  {key:20}: {value:8.3f}")
    print("=" * 60)

    follower_right.send_action(act_right)
    follower_left.send_action(act_left)

    time.sleep(0.02)
