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

"""
This script loads a policy and serves it over gRPC, allowing remote clients to get actions.

Example:
python -m lerobot.scripts.serve_policy --policy lerobot/diffusion_pusht --port 50051
"""

import logging
import pickle
import time
from concurrent import futures
from dataclasses import dataclass

import grpc
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.transport import services_pb2, services_pb2_grpc
from lerobot.common.utils.utils import get_safe_torch_device, init_logging
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig


@dataclass
class ServePolicyConfig:
    policy: PreTrainedConfig
    port: int = 50051
    max_workers: int = 10

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


class PolicyServicer(services_pb2_grpc.PolicyServiceServicer):
    """Provides methods that implement functionality of the policy server."""

    def __init__(self, policy: PreTrainedPolicy):
        self.policy = policy
        self.device = get_safe_torch_device(self.policy.config.device)
        self.policy.to(self.device)
        self.policy.eval()
        logging.info(f"Policy '{policy.config.name}' loaded on device '{self.device}'.")

    def SelectActions(self, request, context):
        try:
            observation = pickle.loads(request.data)
            # TODO(rcadene, aliberts): This is a hacky way to make it work with existing
            # `predict_action` function. This should be improved.
            from lerobot.common.utils.control_utils import predict_action

            action_tensor = predict_action(
                observation,
                self.policy,
                self.device,
                self.policy.config.use_amp,
            )
            action = action_tensor.cpu().numpy()
            
            action_bytes = pickle.dumps(action)
            return services_pb2.ActionMessage(data=action_bytes)
        
        except Exception as e:
            logging.error(f"An error occurred during SelectActions: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error processing request: {e}")
            return services_pb2.ActionMessage()


@parser.wrap()
def serve_policy(cfg: ServePolicyConfig):
    init_logging()
    
    # HACK: We parse again the cli args here to get the pretrained path if there was one.
    policy_path = parser.get_path_arg("policy")
    if policy_path:
        cli_overrides = parser.get_cli_overrides("policy")
        cfg.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
        cfg.policy.pretrained_path = policy_path
        
    policy = make_policy(cfg.policy)
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=cfg.max_workers))
    services_pb2_grpc.add_PolicyServiceServicer_to_server(PolicyServicer(policy), server)
    
    server.add_insecure_port(f"[::]:{cfg.port}")
    server.start()
    logging.info(f"Server started. Listening on port {cfg.port}...")
    
    try:
        while True:
            time.sleep(86400) # One day
    except KeyboardInterrupt:
        logging.info("Shutting down server...")
        server.stop(0)


if __name__ == "__main__":
    serve_policy() 