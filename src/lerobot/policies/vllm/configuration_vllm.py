"""Configuration for the remote vLLM (OpenPI) policy."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.constants import ACTION

# LIBERO (libero_sim / LIBERO_PANDA) modality — the authoritative convention from
# Isaac-GR00T/examples/LIBERO/modality.json. Action is the NATIVE LIBERO 7-D OSC_POSE
# command [x, y, z, roll, pitch, yaw, gripper]; state adds a 2-D gripper qpos (8-D total).
LIBERO_MODALITY_CONFIG: dict = {
    "state": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 8},
    },
    "action": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 7},
    },
}
# Ordered LIBERO action keys -> the 7-D action vector fed to the LIBERO env.
LIBERO_ACTION_KEYS: list[str] = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

# DROID-style modality (legacy / non-LIBERO embodiments).
DROID_MODALITY_CONFIG: dict = {
    "state": {
        "eef_9d": {"start": 0, "end": 9},
        "gripper_position": {"start": 9, "end": 10},
        "joint_position": {"start": 10, "end": 17},
    },
    "action": {
        "eef_9d": {"start": 0, "end": 9},
        "gripper_position": {"start": 9, "end": 10},
        "joint_position": {"start": 10, "end": 17},
    },
}
# Back-compat alias.
DEFAULT_MODALITY_CONFIG = LIBERO_MODALITY_CONFIG


@PreTrainedConfig.register_subclass("vllm")
@dataclass
class VllmConfig(PreTrainedConfig):
    """Config for a policy that proxies inference to a remote vLLM OpenPI policy server.

    The server (e.g. vllm-omni GR00T-N1.7) accepts a *raw* observation over a WebSocket
    (msgpack-numpy) and returns a per-action-key dict of *absolute* action trajectories.
    This policy encodes LeRobot observations into that request and decodes the response
    into the env action space. The default decode is a generic "flat concat" of the keys
    named in ``action_keys`` (so a new embodiment whose action is a flat vector is
    config-only); ``action_space="eef_9d"`` selects an alternative SE(3)/rot6d math path.
    """

    # --- action chunking ---
    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 16

    # --- remote server connection ---
    host: str = "127.0.0.1"
    port: int = 8000
    ws_path: str = "/v1/realtime/robot/openpi"
    connect_timeout_s: float = 30.0
    max_msg_bytes: int = 64 * 1024 * 1024

    # --- request construction ---
    # `libero_sim` is the LIBERO-specific GR00T-N1.7 embodiment (projector id 2).
    embodiment: str = "libero_sim"
    # If set, overrides the env-provided task string as the language prompt.
    prompt_override: str | None = None
    # Fallback image (H, W); overridden by the server handshake metadata when available.
    image_height: int = 256
    image_width: int = 256
    external_camera_obs_key: str = "observation.images.image"  # LIBERO agentview
    wrist_camera_obs_key: str = "observation.images.image2"  # LIBERO eye-in-hand
    send_wrist_camera: bool = True
    # Action/state convention: "libero_7d" (native LIBERO [x,y,z,roll,pitch,yaw,gripper],
    # for the libero_sim/LIBERO_PANDA embodiment) or "eef_9d" (legacy DROID-style).
    action_space: str = "libero_7d"
    # Send images as an ordered list ([external_0, wrist]) rather than a dict. The deployed
    # vllm-omni server feeds `images` straight into the HF image processor, which requires a
    # single image or a list (a dict raises "only a single or a list of entries is supported").
    images_as_list: bool = True
    # LiberoProcessorStep already rotates LIBERO images 180°, so default off here.
    flip_images_180: bool = False
    modality_config: dict = field(default_factory=lambda: DEFAULT_MODALITY_CONFIG)
    # Optional per-request flow-matching seed for reproducibility (None = disabled).
    request_seed: int | None = None

    # --- state encoding ---
    # LeRobot's LiberoProcessorStep flattens state to [eef_pos(3), eef_axisangle(3), gripper_qpos(2)].
    state_obs_key: str = "observation.state"
    # gripper_qpos (2 finger joints) -> single normalized gripper position in [0, 1].
    gripper_qpos_open: float = 0.04  # per-finger qpos at fully open
    gripper_qpos_closed: float = 0.0  # per-finger qpos at fully closed
    # joint positions are not surfaced by LiberoProcessorStep; send zeros unless available.
    send_joint_position: bool = True
    joint_dim: int = 7

    # --- action decoding ---
    # Generic "flat concat" decode (used for every action_space except "eef_9d"): the
    # server returns one trajectory per key; we concatenate the keys named in `action_keys`
    # (in order) into the env action vector. Adding a new embodiment whose action is a flat
    # concat of scalar channels is config-only (set `embodiment` + `action_keys` + the
    # gripper flags below). `eef_9d` keeps its own SE(3)/rot6d math path (see modeling).
    action_keys: list[str] = field(default_factory=lambda: list(LIBERO_ACTION_KEYS))
    # Subset of `action_keys` carrying a gripper command; these get the gripper
    # post-processing below. All other keys are treated as continuous pose channels.
    gripper_keys: list[str] = field(default_factory=lambda: ["gripper"])
    # Must match the env control_mode ("relative" delta or "absolute"). (eef_9d only.)
    control_mode: str = "relative"
    action_scale: float = 1.0
    gripper_action_open: float = -1.0
    gripper_action_close: float = 1.0
    # The vllm-omni server returns ABSOLUTE actions (current state + predicted delta).
    # Isaac-GR00T's native eval feeds the raw DELTA to env.step (relative control), so we
    # subtract the current state (matched by channel index) to recover it. Default True.
    decode_subtract_state: bool = True
    # Gripper post-processing applied to `gripper_keys` (matches Isaac-GR00T's LIBERO eval):
    # normalize [0,1]->[-1,1], then optional sign-binarize, then optional sign-invert.
    gripper_normalize: bool = True
    gripper_binarize: bool = True
    gripper_invert: bool = True
    # Clip the final action to the env's [-1, 1] box.
    clip_action: bool = True

    # Server normalizes/denormalizes; keep the policy-side pipeline identity.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})"
            )
        if self.control_mode not in ("relative", "absolute"):
            raise ValueError(f"control_mode must be 'relative' or 'absolute', got {self.control_mode}")

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}{self.ws_path}"

    @property
    def action_dim(self) -> int:
        """Env action dimension: a flat concat of `action_keys`, except the `eef_9d`
        path which is fixed 7-D (xyz(3) + axisangle(3) + gripper(1))."""
        if self.action_space == "eef_9d":
            return 7
        return len(self.action_keys)

    # --- PreTrainedConfig abstract interface ---
    def validate_features(self) -> None:
        # This policy reads images + a flat state from the env preprocessor and does not
        # build normalization buffers, so we only ensure an ACTION output exists.
        if not self.output_features:
            self.output_features = {
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))
            }

    def get_optimizer_preset(self) -> AdamWConfig:
        # Inference-only policy; provided for API completeness (unused during eval).
        return AdamWConfig()

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
