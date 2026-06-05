"""VllmPolicy: a LeRobot policy that delegates action inference to a remote vLLM
OpenPI WebSocket server (e.g. vllm-omni GR00T-N1.7).

It does not hold model weights. At each step it encodes the (LIBERO) observation into
an OpenPI request, sends it to the remote server, and decodes the returned *absolute*
action trajectory into LIBERO's 7-D action space, buffering ``n_action_steps`` actions.
"""

from __future__ import annotations

import builtins
import logging
import uuid
from collections import deque
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy

from .client import OpenPIClient
from .configuration_vllm import VllmConfig
from .encoding import (
    axisangle_to_matrix,
    chw01_to_hwc_uint8,
    eef_9d_from_pos_axisangle,
    gripper_position_to_action,
    gripper_qpos_to_position,
    matrix_to_axisangle,
    rot6d_to_matrix,
)

logger = logging.getLogger(__name__)
T = TypeVar("T", bound="VllmPolicy")


def _resize_hwc(img: np.ndarray, height: int, width: int) -> np.ndarray:
    if img.shape[0] == height and img.shape[1] == width:
        return img
    try:
        import cv2

        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    except Exception:  # pragma: no cover - fallback nearest-neighbour
        ys = (np.linspace(0, img.shape[0] - 1, height)).astype(int)
        xs = (np.linspace(0, img.shape[1] - 1, width)).astype(int)
        return img[ys][:, xs]


class VllmPolicy(PreTrainedPolicy):
    """Remote-inference policy proxying to a vLLM OpenPI server."""

    name = "vllm"
    config_class = VllmConfig

    def __init__(self, config: VllmConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        # Dummy buffer so the module has a device and `.to(...)`/`.parameters()` behave.
        self.register_buffer("_device_marker", torch.zeros(1), persistent=False)
        self._client = OpenPIClient(
            url=config.ws_url,
            connect_timeout_s=config.connect_timeout_s,
            max_msg_bytes=config.max_msg_bytes,
        )
        self._server_image_hw: tuple[int, int] | None = None
        self.reset()

    # --- lifecycle ---
    def reset(self):
        """Clear the action buffer and start a fresh server session."""
        self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)
        self._session_id = str(uuid.uuid4())

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: VllmConfig | None = None,
        **kwargs,
    ) -> T:
        """Build the policy from config only (no weights to load for a remote policy)."""
        if config is None:
            config = VllmConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        policy = cls(config)
        policy.eval()
        return policy

    def get_optim_params(self) -> dict:
        return self.parameters()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        raise NotImplementedError("VllmPolicy is inference-only (remote server); training is unsupported.")

    # --- inference ---
    @property
    def _device(self) -> torch.device:
        return self._device_marker.device

    def _image_hw(self) -> tuple[int, int]:
        if self._server_image_hw is not None:
            return self._server_image_hw
        return (self.config.image_height, self.config.image_width)

    def _maybe_handshake(self) -> None:
        """Fetch server image resolution once (best-effort)."""
        if self._server_image_hw is not None:
            return
        try:
            meta = self._client.handshake()
            res = meta.get("image_resolution")
            if res and len(res) == 2:
                self._server_image_hw = (int(res[0]), int(res[1]))
        except Exception as exc:  # pragma: no cover - non-fatal
            logger.debug("Server handshake for image resolution failed: %s", exc)
            self._server_image_hw = (self.config.image_height, self.config.image_width)

    def _build_request(self, batch: dict[str, Tensor], i: int) -> dict[str, Any]:
        cfg = self.config
        state = batch[cfg.state_obs_key][i].detach().cpu().numpy().astype(np.float64).reshape(-1)

        if cfg.action_space == "eef_9d":
            # legacy DROID eef_9d: derive xyz+rot6d and a scalar gripper from the flat state.
            eef_9d = eef_9d_from_pos_axisangle(state[0:3], state[3:6])
            gripper_position = gripper_qpos_to_position(
                state[6:8], cfg.gripper_qpos_open, cfg.gripper_qpos_closed
            )
            raw_state = {
                "eef_9d": [float(x) for x in eef_9d],
                "gripper_position": [float(gripper_position)],
            }
            if cfg.send_joint_position:
                raw_state["joint_position"] = [0.0] * cfg.joint_dim
        else:
            # Generic: slice the flat LeRobot state vector into named keys per
            # modality_config["state"]. For LIBERO (LiberoProcessorStep state
            # [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]) this yields
            # {x,y,z,roll,pitch,yaw at single dims, gripper at 2 dims}.
            state_modality = (cfg.modality_config or {}).get("state", {})
            if not state_modality:
                raise RuntimeError(
                    f"action_space={cfg.action_space!r} needs modality_config['state'] to encode "
                    "the request state (a {key: {'start', 'end'}} slicing of observation.state)."
                )
            raw_state = {
                key: [float(v) for v in state[int(spec["start"]) : int(spec["end"])]]
                for key, spec in state_modality.items()
            }

        h, w = self._image_hw()
        ext = chw01_to_hwc_uint8(batch[cfg.external_camera_obs_key][i].detach().cpu().numpy(), cfg.flip_images_180)
        ordered: list[tuple[str, np.ndarray]] = [("external_0", _resize_hwc(ext, h, w))]
        if cfg.send_wrist_camera and cfg.wrist_camera_obs_key in batch:
            wrist = chw01_to_hwc_uint8(
                batch[cfg.wrist_camera_obs_key][i].detach().cpu().numpy(), cfg.flip_images_180
            )
            ordered.append(("wrist", _resize_hwc(wrist, h, w)))
        # The deployed server requires a list (or single image); a dict raises in the HF
        # image processor. `images_as_list` keeps a dict path available for other servers.
        images: Any = [im for _, im in ordered] if cfg.images_as_list else dict(ordered)

        prompt = cfg.prompt_override
        if prompt is None:
            task = batch.get("task")
            prompt = task[i] if isinstance(task, (list, tuple)) and i < len(task) else (task or "")

        obs: dict[str, Any] = {
            "session_id": self._session_id,
            "embodiment": cfg.embodiment,
            "modality_config": cfg.modality_config,
            "prompt": prompt,
            "state": raw_state,
            "images": images,
        }
        if cfg.request_seed is not None:
            obs["seed"] = int(cfg.request_seed)
        return obs

    def _decode_concat(self, actions: dict[str, np.ndarray], state_i: np.ndarray) -> np.ndarray:
        """Generic decode: concatenate the server's per-key trajectories named in
        ``cfg.action_keys`` (in order) into the env action vector ``(n, len(action_keys))``.

        Per channel, in order:
          - optionally de-bias by the current state (``decode_subtract_state``, matched by
            channel index) — the vllm-omni server returns ABSOLUTE values (current state +
            delta), so this recovers the per-step delta for relative-control envs;
          - continuous pose channels are scaled by ``action_scale``;
          - gripper channels (those in ``cfg.gripper_keys``) get gripper post-processing:
            normalize [0,1]->[-1,1] (``gripper_normalize``), optional sign-binarize, invert.

        This reproduces Isaac-GR00T's LIBERO eval for the default LIBERO keys, but is
        embodiment-agnostic: any flat-concat action space works by setting ``action_keys``.
        """
        cfg = self.config
        keys = cfg.action_keys
        missing = [k for k in keys if k not in actions]
        if missing:
            raise RuntimeError(
                f"Server response missing action key(s) {missing}; got {sorted(actions.keys())}"
            )
        horizon = min(int(np.asarray(actions[k]).shape[0]) for k in keys)
        n = min(cfg.n_action_steps, horizon)
        out = np.zeros((n, len(keys)), dtype=np.float32)
        gripper_keys = set(cfg.gripper_keys)

        for idx, key in enumerate(keys):
            col = np.asarray(actions[key], dtype=np.float32).reshape(horizon, -1)[:n, 0]
            if cfg.decode_subtract_state and idx < state_i.shape[0]:
                col = col - np.float32(state_i[idx])
            if key in gripper_keys:
                if cfg.gripper_normalize:
                    col = 2.0 * col - 1.0  # [0,1] -> [-1,1]
                if cfg.gripper_binarize:
                    col = np.sign(col)
                if cfg.gripper_invert:
                    col = -col
            else:
                col = col * cfg.action_scale
            out[:, idx] = col

        if cfg.clip_action:
            np.clip(out, -1.0, 1.0, out=out)
        return out

    def _decode_actions(self, actions: dict[str, np.ndarray], state_i: np.ndarray) -> np.ndarray:
        """Server action dict -> env action chunk (n, action_dim).

        ``eef_9d`` uses an SE(3)/rot6d math path; every other ``action_space`` uses the
        generic flat-concat decode driven by ``cfg.action_keys``.
        """
        if self.config.action_space == "eef_9d":
            return self._decode_eef_9d(actions, state_i)
        return self._decode_concat(actions, state_i)

    def _decode_eef_9d(self, actions: dict[str, np.ndarray], state_i: np.ndarray) -> np.ndarray:
        """DROID-style decode: eef_9d (xyz + rot6d) + gripper_position -> 7-D action."""
        cfg = self.config
        eef_traj = actions.get("eef_9d")
        if eef_traj is None:
            raise RuntimeError(f"Server response missing 'eef_9d'; got keys {sorted(actions.keys())}")
        eef_traj = np.asarray(eef_traj, dtype=np.float64)
        grip_traj = actions.get("gripper_position")
        grip_traj = np.asarray(grip_traj, dtype=np.float64).reshape(-1) if grip_traj is not None else None

        horizon = eef_traj.shape[0]
        n = min(cfg.n_action_steps, horizon)
        out = np.zeros((n, 7), dtype=np.float32)

        prev_xyz = np.asarray(state_i[0:3], dtype=np.float64)
        prev_R = axisangle_to_matrix(state_i[3:6])

        for j in range(n):
            tgt_xyz = eef_traj[j, 0:3]
            tgt_R = rot6d_to_matrix(eef_traj[j, 3:9])
            if cfg.control_mode == "relative":
                out[j, 0:3] = (tgt_xyz - prev_xyz) * cfg.action_scale
                out[j, 3:6] = matrix_to_axisangle(tgt_R @ prev_R.T) * cfg.action_scale
                prev_xyz, prev_R = tgt_xyz, tgt_R
            else:  # absolute
                out[j, 0:3] = tgt_xyz
                out[j, 3:6] = matrix_to_axisangle(tgt_R)
            g = float(grip_traj[j]) if grip_traj is not None and j < grip_traj.shape[0] else 0.0
            out[j, 6] = gripper_position_to_action(g, cfg.gripper_action_open, cfg.gripper_action_close)

        if cfg.clip_action:
            np.clip(out, -1.0, 1.0, out=out)
        return out

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Query the remote server for each env and return (B, n_action_steps, 7)."""
        self._maybe_handshake()
        cfg = self.config
        batch_size = batch[cfg.state_obs_key].shape[0]

        chunks: list[np.ndarray] = []
        for i in range(batch_size):
            obs = self._build_request(batch, i)
            actions, _meta = self._client.infer(obs)
            state_i = batch[cfg.state_obs_key][i].detach().cpu().numpy().astype(np.float64).reshape(-1)
            chunks.append(self._decode_actions(actions, state_i))

        arr = np.stack(chunks, axis=0)  # (B, n, 7)
        return torch.from_numpy(arr).to(self._device)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return one action (B, 7); refill the buffer from the server when empty."""
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)  # (B, n, 7)
            # queue holds n entries, each (B, 7)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
