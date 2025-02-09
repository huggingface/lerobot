import argparse
import asyncio
from contextlib import nullcontext
from datetime import datetime

import msgpack
import numpy as np
import torch
import websockets

from lerobot.common.robot_devices.control_utils import init_policy


async def predict_action(websocket, policy, verbose):
    current_device = "cuda"
    policy = policy.to(current_device)

    with (
        torch.inference_mode(),
        torch.autocast() if use_amp else nullcontext(),
    ):
        try:
            while True:
                packed_message = await websocket.recv()
                start_time = datetime.now()
                observation = msgpack.unpackb(packed_message)
                device = observation.pop("device")
                tensor_observation = {}
                data_obs_items = [
                    (data_key, data_value)
                    for data_key, data_value in observation.items()
                    if "_shape" not in data_key
                ]
                for key, value in data_obs_items:
                    data_shape = observation[key + "_shape"]
                    if "image" in key:
                        raw_data = np.frombuffer(value, dtype=np.uint8).reshape(data_shape)
                        tensor_observation[key] = torch.tensor(raw_data).type(torch.float32) / 255
                        tensor_observation[key] = tensor_observation[key].permute(2, 0, 1).contiguous()
                    else:
                        raw_data = np.frombuffer(value, dtype=np.float32).reshape(data_shape)
                        tensor_observation[key] = torch.tensor(raw_data)
                    tensor_observation[key] = tensor_observation[key].unsqueeze(0)
                    tensor_observation[key] = tensor_observation[key].to(device)
                # Compute the next action with the policy
                if device != current_device:
                    if verbose:
                        print(f"Setting device to: {device}")
                    current_device = device
                    policy = policy.to(current_device)
                action = policy.select_action(tensor_observation)
                # Remove batch dimension and convert to list
                action = action.squeeze(0).to("cpu").numpy()
                end_time = datetime.now()
                if verbose:
                    print("\n*******************************************")
                    print(f"Request received at: {start_time}")
                    print(f"Request processed at: {end_time}")
                    print(f"Processing time: {(end_time - start_time).total_seconds()} seconds\n")
                await websocket.send(action.tobytes())
        except websockets.exceptions.ConnectionClosedOK:
            pass


async def main(policy, port, verbose):
    async with websockets.serve(lambda w: predict_action(w, policy, verbose), "0.0.0.0", port):
        print(f"Server started on port: {port}")
        await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        type=str,
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`."
        ),
    )
    parser.add_argument("-t", "--port", type=int, default=9000, help="Server port")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug messages")
    parser.add_argument(
        "--policy-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()
    pretrained_policy_name = args.pretrained_policy_name_or_path
    policy_overrides = args.policy_overrides
    verbose = args.verbose
    port = args.port

    policy, _, _, use_amp = init_policy(pretrained_policy_name, policy_overrides)
    opt_policy = torch.compile(policy)
    asyncio.run(main(policy, port, verbose))
