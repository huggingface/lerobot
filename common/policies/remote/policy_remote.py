# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import pickle

import grpc
import torch

from lerobot.common.policies.base import BasePolicy
from lerobot.common.transport import services_pb2, services_pb2_grpc

from .config_remote import RemotePolicyConfig


class RemotePolicy(BasePolicy):
    def __init__(self, config: RemotePolicyConfig, dataset_meta=None):
        super().__init__(config, dataset_meta)
        self.config = config
        self.channel = grpc.insecure_channel(f"{config.server_address}:{config.server_port}")
        self.stub = services_pb2_grpc.PolicyServiceStub(self.channel)
        logging.info(f"RemotePolicy connected to server at {config.server_address}:{config.server_port}")

    def select_action(self, observation: dict) -> torch.Tensor:
        try:
            # Serialize the observation dictionary
            observation_bytes = pickle.dumps(observation)
            request = services_pb2.ObservationMessage(data=observation_bytes)

            # Make the RPC call
            response = self.stub.SelectActions(request)

            # Deserialize the received action
            action_array = pickle.loads(response.data)

            # Convert numpy array to torch tensor
            action_tensor = torch.from_numpy(action_array)
            return action_tensor

        except grpc.RpcError as e:
            logging.error(f"gRPC call failed: {e.details()}")
            # Return a zero tensor or handle the error as appropriate
            # For safety, returning a zero action might be best.
            # The shape needs to be determined from metadata if available.
            # This part might need more robust error handling.
            logging.warning("Returning a zero action due to RPC error.")
            # Assuming action is a flat tensor, this is a placeholder.
            # A more robust solution would get the action size from `dataset_meta`.
            return torch.zeros(self.config.action_dim)

    def to(self, device):
        # This policy runs on a remote server, so this is a no-op.
        # The device of the returned tensor is handled by the server.
        # The client side will receive a numpy array and convert it to a tensor.
        logging.info("RemotePolicy `to(device)` called, but computation is remote. This is a no-op.")
        return self

    def eval(self):
        # This policy is always in "eval" mode on the client side.
        pass

    def train(self, mode: bool = True):
        # This policy cannot be trained directly.
        pass
