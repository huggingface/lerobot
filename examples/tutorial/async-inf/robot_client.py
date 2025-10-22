import threading

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import visualize_action_queue_size
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower import SO100FollowerConfig

# these cameras must match the ones expected by the policy - find your cameras with lerobot-find-cameras
# check the config.json on the Hub for the policy you are using to see the expected camera specs
camera_cfg = {
    "up": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
    "side": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
}

# # find ports using lerobot-find-port
follower_port = ...  # something like "/dev/tty.usbmodem58760431631"

# # the robot ids are used the load the right calibration files
follower_id = ...  # something like "follower_so100"

robot_cfg = SO100FollowerConfig(port=follower_port, id=follower_id, cameras=camera_cfg)

server_address = ...  # something like "127.0.0.1:8080" if using localhost

# 3. Create client configuration
client_cfg = RobotClientConfig(
    robot=robot_cfg,
    server_address=server_address,
    policy_device="mps",
    policy_type="act",
    pretrained_name_or_path="fracapuano/robot_learning_tutorial_act",
    chunk_size_threshold=0.5,  # g
    actions_per_chunk=50,  # make sure this is less than the max actions of the policy
)

# 4. Create and start client
client = RobotClient(client_cfg)

# 5. Provide a textual description of the task
task = ...

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
