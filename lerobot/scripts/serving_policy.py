import dataclasses
import enum
import logging
import socket
import tyro

import torch

from lerobot.common.utils.utils import (
    init_logging,
    get_safe_torch_device,
)
from lerobot.common.utils.random_utils import set_seed
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.common.serving.websocket_policy_server import WebsocketPolicyServer
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy


class PolicyType(enum.Enum):
    """Supported environments."""

    ACT = "act"
    DIFFUSION = "diffusion"
    PI0 = "pi0"

@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Checkpoint directory (e.g., "outputs/train/act_move_reel_0322_nodepth/checkpoints/040000/pretrained_mode").
    path: str
    
    # policy type
    type: str

@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""
    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint = dataclasses.field(default_factory=Checkpoint)

def main(args: Args) -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(1000)
    
    print(args)
    
    # policy has been in device and evaluated
    if args.policy.type == PolicyType.ACT.value:
        policy = ACTPolicy.from_pretrained(args.policy.path)
    elif args.policy.type == PolicyType.DIFFUSION.value:
        policy = DiffusionPolicy.from_pretrained(args.policy.path)
    
    # Record the policy's behavior.
    # if args.record:
    #     policy = _policy.PolicyRecorder(policy, "policy_records")

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata={},
    )
    server.serve_forever()


if __name__ == "__main__":
    init_logging()
    main(tyro.cli(Args))
