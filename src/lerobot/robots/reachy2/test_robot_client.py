import threading
from lerobot.scripts.server.configs import RobotClientConfig
from lerobot.robots.reachy2 import Reachy2Robot, Reachy2RobotConfig

from lerobot.scripts.server.robot_client import RobotClient
from lerobot.scripts.server.helpers import visualize_action_queue_size


robot_config = Reachy2RobotConfig(
    ip_address="192.168.0.199",
    id="reachy2-pvt02",
    with_mobile_base=False,
    with_l_arm=False,
    with_neck=False,
    with_antennas=False,
)

# 3. Create client configuration
client_cfg = RobotClientConfig(
    robot=robot_config,
    server_address="localhost:8080",
    policy_device="cuda",
    policy_type="act",
    pretrained_name_or_path="CarolinePascal/pick_and_place_bottle",
    chunk_size_threshold=0.5,
    actions_per_chunk=20,  # make sure this is less than the max actions of the policy
)

# 4. Create and start client
client = RobotClient(client_cfg)

# 5. Specify the task
task = "Don't do anything, stay still"

if client.start():
    # Start action receiver thread
    action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
    action_receiver_thread.start()

    try:
        # Run the control loop
        client.control_loop(task)
    except KeyboardInterrupt:
        client.stop()
        action_receiver_thread.join()
        # (Optionally) plot the action queue size
        visualize_action_queue_size(client.action_queue_size)
