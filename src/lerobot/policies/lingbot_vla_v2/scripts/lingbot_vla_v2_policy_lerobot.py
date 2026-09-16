# LingBot-VLA v2 (lerobot ckpt) x RoboTwin evaluation adapter (Route A thin wrapper)
#
# Architecture: RoboTwin eval client (eval_policy_client_lingbotvla.py, unmodified)
#   --websocket--> this server (deploy/lingbot_vla_v2_policy_lerobot.py)
#   --loads--> the lingbot_vla_v2 checkpoint converted to lerobot (with embedded robot_config + norm_stats)
#
# Differences from the upstream deploy/lingbot_vla_v2_policy.py:
#   the upstream version loads the upstream ckpt format (reads lingbotvla_cli.yaml + the
#   upstream FeatureTransform); this version loads the lerobot format:
#   LingbotVLAV2Policy.from_pretrained + its bundled preprocessor.
#
# Key contracts (verified by reading the source; do not break them):
#   - policy.select_action(batch) expects a batch that has **already been through the
#     preprocessor** (it must contain model keys such as images/img_masks/lang_tokens),
#     not raw obs. So this server must first run obs through the preprocessor built by
#     make_lingbot_vla_v2_pre_post_processors, then feed the result to select_action, and
#     pass the output through the postprocessor (unnormalize + canonical -> raw).
#   - The eval client sends upstream keys: cam_high/cam_left_wrist/cam_right_wrist (HWC uint8)
#     + observation.state (joint_action.vector) + task. This server maps them to the lerobot
#     canonical camera keys, and the preprocessor then handles normalization/tokenization/
#     canonical slot mapping.
#   - Actions: select_action returns in raw space (post-processing has already unnormalized,
#     added back the subtract_state offset, and mapped canonical 55-dim -> robot-specific
#     dims). RoboTwin robotwin.yaml uses subtract_state=False (absolute angles).
#
# Usage:
#   python -m deploy.lingbot_vla_v2_policy_lerobot \
#       --model_path /path/to/lingbot-robotwin-6b-lerobot --port 8006
#   # In another shell, run the RoboTwin eval (the client is unmodified):
#   python experiment/robotwin/eval_policy_client_lingbotvla.py --config <task.yml> --port 8006

import argparse
import os
from typing import Any

import numpy as np
import torch

from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingbotVLAV2Policy
from lerobot.policies.lingbot_vla_v2.processor_lingbot_vla_v2 import (
    make_lingbot_vla_v2_pre_post_processors_from_pretrained,
)
from lerobot.utils.constants import OBS_STATE

try:
    from .websocket_policy_server import WebsocketPolicyServer
except ImportError:
    # This file is designed to be dropped into the deploy/ directory of the upstream checkout
    # (next to the official deploy/lingbot_vla_v2_policy.py, which ships its own
    # websocket_policy_server.py). When run directly or imported from elsewhere, fall back to
    # the package-style path.
    from deploy.websocket_policy_server import WebsocketPolicyServer


# Upstream camera keys sent by the RoboTwin client -> lerobot canonical slots.
# Key names must match the origin_keys of the robot_config used at training time
# (robotwin.yaml: cam_high -> camera_top, etc.).
UPSTREAM_TO_LEROBOT_CAM = {
    "observation.images.cam_high": "observation.images.camera_top",
    "observation.images.cam_left_wrist": "observation.images.camera_wrist_left",
    "observation.images.cam_right_wrist": "observation.images.camera_wrist_right",
}


class LerobotLingbotVLAv2Server:
    """RoboTwin policy server that loads a lerobot ckpt.

    Interface aligned with the upstream LingbotVLAv2Server: infer(obs) -> {"action": np.ndarray}.
    - Input: obs in the RoboTwin client's upstream format (images HWC uint8,
      state=joint_action.vector, task=str).
    - Output: {"action": robot raw joint angles as np}, shape (action_dim,), absolute angles
      (robotwin profile).
    - chunk: select_action maintains an internal action queue (policy._queues) and pops one
      frame per step; reset clears it.
    """

    def __init__(self, model_path: str, device: str = "cuda", use_bf16: bool = True) -> None:
        self.device = device
        self.use_bf16 = use_bf16
        self._load(model_path)

    def _load(self, model_path: str) -> None:
        print(f"[lerobot-server] loading lerobot ckpt: {model_path}")
        self.model_path = model_path
        self.policy = LingbotVLAV2Policy.from_pretrained(model_path)
        self.policy.to(self.device)
        if self.use_bf16:
            self.policy.to(torch.bfloat16)
        self.policy.eval()
        # The preprocessor / postprocessor are loaded from the ckpt (robot_config +
        # norm_stats were embedded at conversion time), so normalization stays consistent
        # between training and inference.
        self.preprocessor, self.postprocessor = (
            make_lingbot_vla_v2_pre_post_processors_from_pretrained(
                self.policy.config, model_path
            )
        )
        # Robot action dims: the ckpt's output_features.action is the canonical 55; the actual
        # raw dims are sliced out by the postprocessor (unapply) according to the robot_config
        # slots. reset() follows the config embedded in the ckpt.
        self.policy.reset()
        self.global_step = 0

    def reset(self, path_to_pi_model: str | None = None) -> None:
        if path_to_pi_model and path_to_pi_model != self.model_path:
            self._load(path_to_pi_model)
            return
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        self.global_step = 0

    def _to_raw_lerobot_frame(self, observation: dict[str, Any]) -> dict[str, Any]:
        """RoboTwin upstream obs -> lerobot raw obs (dataset frame format, not yet passed
        through the processor).

        Images are converted to HWC uint8 torch; key names are mapped to the canonical camera
        slots; state/task are passed through as-is. The preprocessor then uniformly handles
        normalization/tokenization/canonical slot mapping/adding the batch dimension.
        """

        out: dict[str, Any] = {}
        for up_key, lerobot_key in UPSTREAM_TO_LEROBOT_CAM.items():
            if up_key in observation:
                img = np.ascontiguousarray(observation[up_key])  # HWC uint8
                out[lerobot_key] = torch.from_numpy(img)
        if "observation.state" in observation:
            out[OBS_STATE] = torch.from_numpy(
                np.asarray(observation["observation.state"], dtype=np.float32)
            )
        if "task" in observation:
            out["task"] = observation["task"]
        return out

    @torch.no_grad()
    def infer(self, observation: dict[str, Any], **_: Any) -> dict[str, Any]:
        if observation.get("reset"):
            self.reset(path_to_pi_model=observation.get("path_to_pi_model"))
            return {"action": None}

        raw_frame = self._to_raw_lerobot_frame(observation)
        # Full pipeline: raw obs -> preprocessor -> select_action -> postprocessor -> raw action.
        batch = self.preprocessor(raw_frame)
        action = self.policy.select_action(batch)   # canonical space, already de-normed
        action = self.postprocessor(action)          # canonical 55 -> robot raw joint angles
        if isinstance(action, torch.Tensor):
            action = action.float().cpu().numpy()
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        self.global_step += 1
        return {"action": action}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(description="LingBot-VLA v2 (lerobot ckpt) RoboTwin policy server")
    parser.add_argument("--model_path", type=str, required=True, help="Directory of the lerobot-converted ckpt")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_bf16", type=str2bool, default=True)
    args = parser.parse_args()

    model = LerobotLingbotVLAv2Server(args.model_path, device=args.device, use_bf16=args.use_bf16)
    WebsocketPolicyServer(model, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
