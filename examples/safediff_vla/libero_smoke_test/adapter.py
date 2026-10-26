"""LIBERO / LIBERO-Plus-only smoke-test adapter (see `../libero_transfer_audit_report.md`).

Deliberately NOT registered in `lerobot.action_semantics` (the global contract registry) --
this module exists to (a) route camera keys so a single policy config works unmodified across
`lerobot/libero` and `lerobot/libero_plus`, and (b) hold the *candidate* action-semantics
constants this audit measured empirically, for use ONLY by this smoke test's own rollout-video
diagnostics (`rollout_diagnostics.py`). These constants are NOT wired into training or into
`SafeDiffVLAPolicy`/`find_grasp_target` -- training and eval both consume the dataset's/env's
native 7-D action as-is (LeRobot's usual MEAN_STD-normalize / unnormalize path), which is scale-
and frame-convention-agnostic by construction. The values here are read by the diagnostic script
to annotate rollout videos ("predicted delta-xyz", "predicted open/close") so a human/the agent
reviewing the video can judge whether the *audit's* candidate semantics actually explain what the
robot does on screen -- they are promoted to a real `find_grasp_target` polarity/threshold fix
only after that video review confirms them (see the smoke-test report).

Empirical basis (from the dataset/action-semantics audit, `lerobot/libero` + `lerobot/libero_plus`
`meta/*.parquet`, real HF data, not VLABench's 7-D-absolute assumption):
  - action is a DELTA command, not absolute (translation corr=0.93-0.96, world-frame rotation
    corr=0.89-0.93 vs `Δobservation.state`).
  - translation: empirical best-fit scale ~0.011 m/unit (registry's `LIBERO_ACTION_CONTRACT` says
    0.05 -- ~4.5x off; NOT corrected globally per this task's instructions).
  - rotation: must be composed in the WORLD frame (`R_new = R_delta(action) @ R_old`), not body
    frame (body-frame composition gives corr=-0.60 to -0.65, world-frame gives +0.89 to +0.93).
    Empirical best-fit scale ~0.10 rad/unit (registry says 0.5 -- ~5x off).
  - gripper: LOW-CONFIDENCE candidate polarity is action>0 ~ close, action<0 ~ open (weak raw-qpos
    correlation, -0.10 to -0.17; consistent with `envs/libero.py`'s `get_libero_dummy_action()`
    using gripper=-1 as the rest/no-op value). This is the opposite of VLABench's convention
    baked into `utils._action_gripper_is_open` (value>0.5 => open), which is why
    `find_grasp_target` must NOT be reused unmodified on LIBERO until this is confirmed on video.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lerobot.processor.pipeline import ObservationProcessorStep


@dataclass(frozen=True)
class LiberoSmokeTestActionSemantics:
    """Candidate (UNCONFIRMED) action semantics for one LIBERO-family dataset.

    `gripper_polarity_confidence` is deliberately a free-text flag, not a bool -- there is no
    "confirmed" state until a human/the agent has looked at rollout video and matched predicted
    gripper sign against observed finger motion for this exact dataset.
    """

    name: str
    fps: float
    translation_scale_candidate: tuple[float, float, float]
    rotation_scale_candidate: tuple[float, float, float]
    rotation_composition: str  # "world" -- see module docstring
    gripper_open_is_negative_candidate: bool
    gripper_polarity_confidence: str  # "low" until video-confirmed


LIBERO_SMOKE_ACTION_SEMANTICS = LiberoSmokeTestActionSemantics(
    name="libero",
    fps=10.0,
    translation_scale_candidate=(0.011, 0.011, 0.011),
    rotation_scale_candidate=(0.10, 0.10, 0.10),
    rotation_composition="world",
    gripper_open_is_negative_candidate=True,
    gripper_polarity_confidence="low",
)

LIBERO_PLUS_SMOKE_ACTION_SEMANTICS = LiberoSmokeTestActionSemantics(
    name="libero_plus",
    fps=20.0,
    translation_scale_candidate=(0.011, 0.011, 0.011),
    rotation_scale_candidate=(0.10, 0.10, 0.10),
    rotation_composition="world",
    gripper_open_is_negative_candidate=True,
    gripper_polarity_confidence="low",
)


# Camera-key routing only (no scale/semantics implications).
#
# `TrainPipelineConfig.rename_map` turns out NOT to apply here: it only rewrites dataset keys to
# match an ALREADY-PRETRAINED checkpoint's fixed input_features (`configs/train.py`'s own
# validation refuses it for a fresh/from-scratch policy -- "Fresh initialization derives feature
# names from the current dataset"). Since this smoke test trains fresh (no `pretrained_path`),
# each policy's `input_features` are simply whatever the dataset's own keys are: `image`/`image2`
# for `lerobot/libero`, `front`/`wrist` for `lerobot/libero_plus` -- both train fine as-is, no
# training-side adapter needed.
#
# The real mismatch is EVAL-time only: `LiberoPlusEnv` inherits `LiberoEnv`'s `camera_name_mapping`
# default unchanged (`envs/configs.py` -- confirmed by reading the class, `LiberoPlusEnv` adds no
# override), so `env.type=libero_plus` emits `image`/`image2` by default, NOT `front`/`wrist` --
# it would silently hand a `front`/`wrist`-trained policy the wrong keys. Fix: pass this mapping
# as `LiberoPlusEnv(camera_name_mapping=...)` at eval time only (`rollout_diagnostics.py`), using
# the parameter `LiberoEnv`/`LiberoPlusEnv` already expose for exactly this -- no change to
# `envs/libero.py` or `envs/configs.py`.
EVAL_CAMERA_NAME_MAPPING = {
    "libero": None,  # env default (image/image2) already matches the dataset -- no override
    "libero_plus": {"agentview_image": "front", "robot0_eye_in_hand_image": "wrist"},
}


# Task text this smoke test uses, and its dataset-native `task_index` per HF dataset (only valid
# for joining episode subsets against `meta/tasks.parquet` -- see audit's task-index-misalignment
# finding). This is NOT the LIBERO benchmark suite's own per-suite task id used by `env.task_ids`
# (that mapping is looked up separately, from the installed `libero` package's own benchmark dict,
# by task text -- see `run_smoke_test.py`).
SMOKE_TEST_TASK_TEXT = "pick up the milk and place it in the basket"
SMOKE_TEST_SUITE = "libero_object"
SMOKE_TEST_SUITE_TASK_ID = 7  # verified against `benchmark.get_benchmark_dict()["libero_object"]().tasks[7].language`


# ---------------------------------------------------------------------------------------------
# SafeDiff-VLA-only 8D -> 7D state adapter (LIBERO family only).
#
# `SafeDiffVLAConfig.validate_features` (`temporal_decoder`/siblings) hard-requires
# `observation.state` to be 7-D matching action's `[xyz, rx, ry, rz, gripper]` layout, because
# `rotation_encoding.py`'s sin/cos conversion slices both state and action with the SAME fixed
# indices. That is a real constraint for VLABench (state and action already share this 7-D
# layout there) but LIBERO/LIBERO-Plus's real `observation.state` is 8-D: eef_xyz[3] + rotvec[3]
# + TWO gripper-finger qpos values (see the dataset/action-semantics audit). SmolVLA has no such
# constraint and needs none of this -- only SafeDiff-VLA's temporal_decoder path does.
#
# Fix, scoped to LIBERO only, never touching `configuration_safediff_vla.py`/`modeling_safediff_vla.py`
# or the global registry: collapse the two finger-qpos dims into their mean (a parallel gripper's
# two fingers move symmetrically, so the mean is a faithful 1-D opening-width proxy, not an
# arbitrary drop) wherever `observation.state` is produced for lerobot/libero(_plus) --
# implemented as process-local monkeypatches (this script's own process only; no repository file
# is modified) for the TRAINING path, and as an explicit extra `ObservationProcessorStep` for the
# EVAL/rollout path.
# ---------------------------------------------------------------------------------------------

LIBERO_FAMILY_REPO_IDS = {"lerobot/libero", "lerobot/libero_plus"}


def _collapse_gripper_last_dim(state_like):
    """`[..., 8]` (eef_xyz, rotvec, finger1_qpos, finger2_qpos) -> `[..., 7]`.

    Uses `(finger1 - finger2) / 2`, NOT the mean. Real data (`grasp_transition_audit.py`'s
    per-episode traces): the two fingers are near-antisymmetric (open ~= [+0.040, -0.040], closed
    ~= [+0.028, -0.026]) -- their MEAN is therefore ~0 in BOTH states (open mean ~0.00005, closed
    mean ~0.0012), destroying the actual open/closed signal almost entirely. The HALF-DIFFERENCE
    preserves it near-losslessly (open ~0.040, closed ~0.027, cleanly separated) since it's
    algebraically ~= finger1 alone modulo the small asymmetry. This was corrected FROM the mean
    after the mean version produced the "LOW confidence" gripper-state reading in the original
    dataset/action-semantics audit -- that low confidence was an artifact of the collapse method,
    not the underlying data (see `grasp_transition_audit.py` for the corrected, HIGH-confidence
    read: action[6] is exactly binary {-1, +1}, -1=open/+1=close, and state, using this
    half-difference, cleanly separates open (~0.040) from closed (~0.027-0.030)).
    No-op if the last dim is already != 8.
    """
    import numpy as np
    import torch

    is_torch = isinstance(state_like, torch.Tensor)
    arr = state_like.detach().cpu().numpy() if is_torch else np.asarray(state_like)
    if arr.shape[-1] != 8:
        return state_like
    half_diff = (arr[..., 6] - arr[..., 7]) / 2.0
    collapsed = np.concatenate([arr[..., :6], half_diff[..., None]], axis=-1).astype(arr.dtype, copy=False)
    if is_torch:
        return torch.as_tensor(collapsed, dtype=state_like.dtype, device=state_like.device)
    return collapsed


def install_libero_gripper_collapse_training_patch() -> None:
    """Monkeypatch dataset stats loading, policy-feature derivation, and the training loop's
    dataset-batch preprocessing so the 8-D LIBERO family state is 7-D everywhere
    `SafeDiffVLAConfig`/`SafeDiffVLAPolicy` looks at it, WITHOUT touching the declared
    `meta/info.json` feature shape -- that shape also drives the raw HF
    `datasets.Dataset.from_parquet(..., features=...)` load of the actual on-disk data (still
    genuinely 8-D), and declaring it 7 there raises a pyarrow cast error (tried first, failed:
    "Couldn't cast array of type list<float> to List(float32, length=7)"). So the raw load keeps
    seeing 8-D (correct, matches disk), and this patches:
      - `lerobot.policies.factory.dataset_to_policy_features` -- what `make_policy` uses to build
        `cfg.input_features['observation.state']`, so `SafeDiffVLAConfig.validate_features()` sees
        7 and the model is actually built with a 7-wide state input layer.
      - `load_stats` -- normalization stats must match the 7-D tensor the model will see.
      - `lerobot.scripts.lerobot_train._preprocess_dataset_batch` -- collapses the ALREADY-
        COLLATED batch (shape `[B, 8]` -> `[B, 7]`) right before it enters the normalizer. This
        runs in the training loop's main process regardless of `--num-workers`, unlike an earlier
        version of this patch that instead wrapped `LeRobotDataset.__getitem__` -- that ran
        per-worker-process and silently did nothing under the default `dataloader_
        multiprocessing_context="spawn"` (each spawned worker re-imports fresh, unpatched modules;
        confirmed by a real crash with `--num-workers 4`: normalizer tried to broadcast an
        un-collapsed 8-D batch against 7-D stats). Patching the batch-level function instead makes
        this correct at ANY `--num-workers`, and deliberately does NOT force
        `dataloader_multiprocessing_context="fork"` to get there -- LeRobot defaults to "spawn"
        specifically because LIBERO's video backend (PyAV/torchcodec/ffmpeg) is not fork-safe, and
        forcing fork to fix an 8D/7D mismatch would trade a correctness bug for a real risk of
        worker deadlocks/corruption.
    Only applied when explicitly called (see `train_smoke.py`'s and the FINAL training scripts'
    `--policy safediff_vla` branch) -- SmolVLA runs never call this.
    """
    import numpy as np

    from lerobot.datasets import dataset_metadata as dm_module
    from lerobot.datasets import io_utils as io_module
    from lerobot.policies import factory as policy_factory_module
    from lerobot.scripts import lerobot_train as train_module

    original_load_stats = io_module.load_stats
    original_preprocess_batch = train_module._preprocess_dataset_batch
    original_dataset_to_policy_features = policy_factory_module.dataset_to_policy_features

    def patched_dataset_to_policy_features(features):
        policy_features = original_dataset_to_policy_features(features)
        state_ft = policy_features.get("observation.state")
        if state_ft is not None and tuple(state_ft.shape) == (8,):
            policy_features["observation.state"] = type(state_ft)(type=state_ft.type, shape=(7,))
        return policy_features

    def patched_load_stats(local_dir):
        stats = original_load_stats(local_dir)
        if stats and "observation.state" in stats:
            s = stats["observation.state"]
            for stat_name, value in list(s.items()):
                value = np.asarray(value)
                if value.shape[-1] == 8:
                    # Approximate (no stored covariance between finger1/finger2 to compute this
                    # exactly): treat them as independent for std propagation. Only feeds
                    # MEAN_STD input normalization, not the gripper-transition detector itself
                    # (that reads raw un-normalized values) -- doesn't need to be exact, just a
                    # reasonable scale.
                    if stat_name == "mean":
                        collapsed = np.concatenate(
                            [value[..., :6], ((value[..., 6] - value[..., 7]) / 2.0)[..., None]], axis=-1
                        )
                    elif stat_name == "std":
                        half_diff_std = np.sqrt(value[..., 6] ** 2 + value[..., 7] ** 2) / 2.0
                        collapsed = np.concatenate([value[..., :6], half_diff_std[..., None]], axis=-1)
                    elif stat_name == "min":
                        half_diff_min = np.atleast_1d((value[..., 6] - value[..., 7]) / 2.0)
                        collapsed = np.concatenate([value[..., :6], half_diff_min], axis=-1)
                    elif stat_name == "max":
                        half_diff_max = np.atleast_1d((value[..., 6] - value[..., 7]) / 2.0)
                        collapsed = np.concatenate([value[..., :6], half_diff_max], axis=-1)
                    else:
                        collapsed = value
                    s[stat_name] = collapsed
        return stats

    def patched_preprocess_batch(batch, camera_keys, rename_map, preprocessor):
        if "observation.state" in batch and batch["observation.state"].shape[-1] == 8:
            batch = dict(batch)
            batch["observation.state"] = _collapse_gripper_last_dim(batch["observation.state"])
        return original_preprocess_batch(batch, camera_keys, rename_map, preprocessor)

    io_module.load_stats = patched_load_stats
    dm_module.load_stats = patched_load_stats
    policy_factory_module.dataset_to_policy_features = patched_dataset_to_policy_features
    train_module._preprocess_dataset_batch = patched_preprocess_batch


@dataclass
class GripperCollapseObservationStep(ObservationProcessorStep):
    """Eval/rollout-side counterpart of `install_libero_gripper_collapse_training_patch`: an
    `ObservationProcessorStep` (same base class as `RenameObservationsProcessorStep`) that
    collapses `observation.state`'s last 2 dims (8->7) the same way, inserted AFTER
    `LiberoProcessorStep` in a custom `PolicyProcessorPipeline` built directly in
    `rollout_diagnostics.py` (not via `LiberoEnv.get_env_processors()`, which stays unmodified).
    """

    def observation(self, observation):
        from lerobot.utils.constants import OBS_STATE

        obs = dict(observation)
        if OBS_STATE in obs:
            obs[OBS_STATE] = _collapse_gripper_last_dim(obs[OBS_STATE])
        return obs

    def get_config(self) -> dict[str, Any]:
        return {}

    def transform_features(self, features):
        from lerobot.configs import PipelineFeatureType
        from lerobot.utils.constants import OBS_STATE

        new_features = features.copy()
        obs_features = dict(features[PipelineFeatureType.OBSERVATION])
        if OBS_STATE in obs_features:
            feat = obs_features[OBS_STATE]
            if tuple(feat.shape) == (8,):
                obs_features[OBS_STATE] = type(feat)(type=feat.type, shape=(7,))
        new_features[PipelineFeatureType.OBSERVATION] = obs_features
        return new_features


# ---------------------------------------------------------------------------------------------
# LIBERO-scoped grasp-transition (gripper polarity/threshold) adapter for SafeDiff-VLA Final.
#
# `utils.find_grasp_target`/`_state_gripper_is_open`/`_action_gripper_is_open` in
# `lerobot/policies/safediff_vla/` hard-code VLABench's convention (action>0.5=open,
# state<0.5=open) -- confirmed WRONG-DIRECTION for LIBERO by direct inspection of real per-episode
# traces (`grasp_transition_audit.py`): LIBERO's `action[..., 6]` is exactly BINARY {-1.0, +1.0}
# across every episode checked (10 episodes of "pick up the milk and place it in the basket" +
# 8 more spread across 4 other tasks, all clean), transitioning -1 -> +1 exactly once per episode
# at a well-defined step, matching `envs/libero.py`'s own `get_libero_dummy_action()` using -1 as
# the rest/no-op (=open) value. `observation.state[..., 6:8]` (raw, pre-collapse) directly tracks
# this: two fingers near [+0.040, -0.040] while open, contracting to ~[+0.028, -0.026] once closed
# on an object -- see `_collapse_gripper_last_dim`'s docstring for why the HALF-DIFFERENCE (not
# mean) of these two values is what actually reaches index 6 of the 7-D collapsed state SafeDiff-
# VLA sees.
#
# Video-verified (see the smoke-test session's transcript / `grasp_transition_frames/`): for 3
# independent episodes (task "pick up the milk...") the transition step detected by THIS adapter's
# threshold/polarity lands exactly where the real recorded video shows the gripper closing on the
# target object -- confirmed by extracting real dataset video frames (not simulation) 1s before,
# at, and 1s after each detected transition index.
LIBERO_ACTION_GRIPPER_OPEN_THRESHOLD = 0.0
LIBERO_STATE_GRIPPER_OPEN_THRESHOLD = 0.034  # collapsed half-diff: ~0.040 open, ~0.027-0.030 closed


def _libero_action_gripper_is_open(values, threshold: float = LIBERO_ACTION_GRIPPER_OPEN_THRESHOLD):
    """LIBERO action[6] convention: -1.0=open, +1.0=close (opposite of VLABench's >threshold=open)."""
    return values <= threshold


def _libero_state_gripper_is_open(values, threshold: float = LIBERO_STATE_GRIPPER_OPEN_THRESHOLD):
    """LIBERO collapsed state[6] (half-diff of the two raw finger qpos) convention: larger=open."""
    return values > threshold


def libero_find_grasp_target(current_state, action, gripper_index: int = 6, gripper_open_threshold=None, action_is_pad=None):
    """LIBERO-scoped reimplementation of `utils.find_grasp_target`'s exact transition-finding
    algorithm, using LIBERO's own gripper polarity/threshold instead of VLABench's. `utils.
    find_grasp_target` itself is never called or modified -- see `install_libero_grasp_target_patch`
    for how this is wired in only for LIBERO-family training, without touching the shared function
    VLABench training still uses. `gripper_open_threshold` is accepted only for call-signature
    compatibility with `find_grasp_target` (SafeDiffVLAPolicy passes `self.config.
    grounded_grasp_gripper_open_threshold`, VLABench-scaled and irrelevant here) and is ignored;
    LIBERO's own fixed, empirically-derived thresholds above are used unconditionally.
    """
    import torch

    state_open_now = _libero_state_gripper_is_open(current_state[..., gripper_index])
    action_open = _libero_action_gripper_is_open(action[..., gripper_index])
    is_open_seq = torch.cat((state_open_now.unsqueeze(1), action_open), dim=1)
    transitions = is_open_seq[:, :-1] & ~is_open_seq[:, 1:]
    if action_is_pad is not None:
        transitions = transitions & ~action_is_pad
    has_transition = transitions.any(dim=1)
    first_transition_idx = transitions.to(torch.float32).argmax(dim=1)
    valid_mask = has_transition & state_open_now
    batch_idx = torch.arange(action.shape[0], device=action.device)
    target_xyz = action[batch_idx, first_transition_idx, :3]
    return target_xyz, valid_mask


def install_libero_grasp_target_patch() -> None:
    """LIBERO-scoped find_grasp_target/gripper-polarity adapter for SafeDiff-VLA Final
    (architecture="temporal_decoder_grounded_grasp_v2_5") on `lerobot/libero`/`lerobot/libero_plus`.

    Patches ONLY the names as seen from `modeling_safediff_vla`'s own module namespace (it
    from-imports `find_grasp_target`/`_state_gripper_is_open`/`_action_gripper_is_open` from
    `.utils`, and also defines `_first_grasp_transition_idx` locally, which calls the same two
    helpers by name -- Python resolves both via THIS module's globals at call time, so patching
    them here transparently covers all of: the 4 direct `find_grasp_target(...)` call sites, the
    `~_state_gripper_is_open(...)`/`~_action_gripper_is_open(...)` phase/polarity checks, and
    `_first_grasp_transition_idx`'s own LOCAL_GRASP-window gating). `lerobot.policies.safediff_vla.
    utils.find_grasp_target` and its two helpers are NEVER modified -- VLABench training through
    the same module in a separate process is completely unaffected.
    """
    from lerobot.policies.safediff_vla import modeling_safediff_vla as model_module

    model_module.find_grasp_target = libero_find_grasp_target
    model_module._state_gripper_is_open = _libero_state_gripper_is_open
    model_module._action_gripper_is_open = _libero_action_gripper_is_open
