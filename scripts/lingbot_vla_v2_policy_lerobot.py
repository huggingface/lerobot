# LingBot-VLA v2(lerobot ckpt)× RoboTwin 评测适配层(路 A 薄封装)
#
# 结构:RoboTwin eval client(eval_policy_client_lingbotvla.py,不改)
#   --websocket--> 本 server(deploy/lingbot_vla_v2_policy_lerobot.py)
#   --加载--> lerobot 转出的 lingbot_vla_v2 checkpoint(含内嵌 robot_config+norm_stats)
#
# 与上游 deploy/lingbot_vla_v2_policy.py 的区别:
#   上游版加载上游 ckpt 格式(读 lingbotvla_cli.yaml + 上游 FeatureTransform);
#   本版加载 lerobot 格式:LingbotVLAV2Policy.from_pretrained + 自带 preprocessor。
#
# 关键契约(读源码确认,别踩):
#   - policy.select_action(batch) 期望 batch **已过 preprocessor**(要有
#     images/img_masks/lang_tokens 等模型键),不是 raw obs。所以本 server 必须先
#     用 make_lingbot_vla_v2_pre_post_processors 建的 preprocessor 处理 obs,
#     再喂 select_action,输出再过 postprocessor(unnormalize+canonical→raw)。
#   - 评测 client 发上游键:cam_high/cam_left_wrist/cam_right_wrist(HWC uint8)+
#     observation.state(joint_action.vector)+ task。本 server 映到 lerobot
#     canonical 相机键,再由 preprocessor 做归一化/tokenize/canonical 槽位映射。
#   - 动作:select_action 返回 raw 空间(后处理已 unnormalize + subtract_state 加回 +
#     canonical 55 维→本体维)。RoboTwin robotwin.yaml 是 subtract_state=False(绝对角)。
#
# 用法:
#   python -m deploy.lingbot_vla_v2_policy_lerobot \
#       --model_path /path/to/lingbot-robotwin-6b-lerobot --port 8006
#   # 另开 shell 跑 RoboTwin eval(client 不改):
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
    # 本文件设计为放进上游 checkout 的 deploy/ 目录(与官方 deploy/lingbot_vla_v2_policy.py
    # 并列,那里自带 websocket_policy_server.py)。直接运行/从别处 import 时回退到包式路径。
    from deploy.websocket_policy_server import WebsocketPolicyServer


# RoboTwin client 端的上游相机键 -> lerobot canonical 槽位。
# 键名必须与训练时 robot_config 的 origin_keys 一致(robotwin.yaml:cam_high→camera_top 等)。
UPSTREAM_TO_LEROBOT_CAM = {
    "observation.images.cam_high": "observation.images.camera_top",
    "observation.images.cam_left_wrist": "observation.images.camera_wrist_left",
    "observation.images.cam_right_wrist": "observation.images.camera_wrist_right",
}


class LerobotLingbotVLAv2Server:
    """加载 lerobot ckpt 的 RoboTwin policy server。

    接口对齐上游 LingbotVLAv2Server:infer(obs) -> {"action": np.ndarray}。
    - 输入:RoboTwin client 的上游格式 obs(图像 HWC uint8,state=joint_action.vector,task=str)。
    - 输出:{"action": 本体 raw 关节角 np},形状 (action_dim,),绝对角(robotwin profile)。
    - chunk:select_action 内部维护动作队列(policy._queues),每步弹一帧;reset 清空。
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
        # preprocessor / postprocessor 从 ckpt 加载(convert 时已把 robot_config +
        # norm_stats 内嵌),保证训练/推理归一化一致。
        self.preprocessor, self.postprocessor = (
            make_lingbot_vla_v2_pre_post_processors_from_pretrained(
                self.policy.config, model_path
            )
        )
        # 本体 action 维:ckpt 的 output_features.action 就是 canonical 55;实际 raw 维
        # 由 postprocessor(unapply)按 robot_config 槽位切出。reset 时按 ckpt 内嵌配置。
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
        """RoboTwin 上游 obs -> lerobot raw obs(未过 processor 的 dataset 帧格式)。

        图像转 HWC uint8 torch;键名按 canonical 相机槽位映射;state/task 原样。
        之后由 preprocessor 统一做归一化/tokenize/canonical 槽位映射/加 batch 维。
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
        # 全链路:raw obs -> preprocessor -> select_action -> postprocessor -> raw action。
        batch = self.preprocessor(raw_frame)
        action = self.policy.select_action(batch)   # canonical 空间、已 de-norm
        action = self.postprocessor(action)          # canonical 55 -> 本体 raw 关节角
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
    parser.add_argument("--model_path", type=str, required=True, help="lerobot 转出的 ckpt 目录")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_bf16", type=str2bool, default=True)
    args = parser.parse_args()

    model = LerobotLingbotVLAv2Server(args.model_path, device=args.device, use_bf16=args.use_bf16)
    WebsocketPolicyServer(model, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
