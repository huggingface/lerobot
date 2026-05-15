#!/usr/bin/env python
"""Hot-prompt switching demo with a real SmolVLA checkpoint.

No physical robot required.  Synthetic (zero) camera frames and robot state
are fed to the real policy so you can see how the action output changes when
you switch the language task live.

Usage
-----
    conda run -n lerobot_rollout python examples/hot_prompt_smolvla_demo.py \\
        --checkpoint /path/to/pretrained_model

    # override task, device, FPS:
    conda run -n lerobot_rollout python examples/hot_prompt_smolvla_demo.py \\
        --checkpoint /path/to/pretrained_model \\
        --task "pick up the bottle" \\
        --device cuda \\
        --fps 1

    # pipe tasks in from another terminal via a FIFO:
    mkfifo /tmp/robot_task
    conda run -n lerobot_rollout python examples/hot_prompt_smolvla_demo.py \\
        --checkpoint /path/to/pretrained_model --fps 1 < /tmp/robot_task &
    echo "grab the cup" > /tmp/robot_task

Press Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hot_prompt_smolvla")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(requested: str) -> str:
    """Return *requested* device, but fall back to CPU if CUDA is unavailable."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU (inference will be slow)")
        return "cpu"
    return requested


def _load_checkpoint_config(checkpoint_path: str) -> dict:
    cfg_file = Path(checkpoint_path) / "config.json"
    with open(cfg_file) as f:
        return json.load(f)


def _build_fake_obs(input_features: dict) -> dict[str, np.ndarray]:
    """Return a dict of zero numpy arrays matching the checkpoint's input_features.

    - STATE features  → float32 zeros, shape (dim,)
    - VISUAL features → uint8 zeros, shape (H, W, C)  [H,W,C required by
                        prepare_observation_for_inference before permutation]
    """
    obs: dict[str, np.ndarray] = {}
    for key, feat in input_features.items():
        ftype = feat.get("type", "")
        shape = feat["shape"]
        if ftype == "VISUAL":
            # config stores (C, H, W); convert to (H, W, C) for the pipeline
            c, h, w = shape
            obs[key] = np.zeros((h, w, c), dtype=np.uint8)
        else:
            obs[key] = np.zeros(shape, dtype=np.float32)
    return obs


def _load_processors(checkpoint_path: str, device: str):
    """Load serialised preprocessor and postprocessor from the checkpoint dir.

    ``from_pretrained`` does not persist the ``to_transition`` / ``to_output``
    callables (they are not JSON-serialisable), so the postprocessor reverts to
    the default ``batch_to_transition`` which expects a dict.  SmolVLA's
    postprocessor was created with ``policy_action_to_transition`` (tensor →
    EnvTransition), so we restore that manually after loading.
    """
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=checkpoint_path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        overrides={"device_processor": {"device": device}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=checkpoint_path,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        overrides={"device_processor": {"device": "cpu"}},
    )
    # Restore the correct tensor ↔ transition converters for the postprocessor.
    postprocessor.to_transition = policy_action_to_transition
    postprocessor.to_output = transition_to_policy_action
    return preprocessor, postprocessor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the pretrained_model directory (contains config.json + model.safetensors)",
    )
    parser.add_argument("--task", default="pick up the bottle", help="Initial task string")
    parser.add_argument("--fps", type=float, default=1.0, help="Control loop frequency (Hz)")
    parser.add_argument("--duration", type=float, default=0.0, help="Run N seconds (0 = infinite)")
    parser.add_argument("--device", default="cuda", help="Torch device: cuda or cpu")
    parser.add_argument(
        "--flush_on_switch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Flush the precomputed action chunk when the task changes so the policy "
            "re-runs the VLM immediately (default: on). "
            "Use --no-flush_on_switch to let the current chunk drain first."
        ),
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)

    # ------------------------------------------------------------------
    # 1. Load checkpoint config
    # ------------------------------------------------------------------
    logger.info("Reading checkpoint config from %s", args.checkpoint)
    ckpt_cfg = _load_checkpoint_config(args.checkpoint)
    input_features = ckpt_cfg["input_features"]
    output_features = ckpt_cfg["output_features"]

    # Build dataset_features + ordered_action_keys expected by make_robot_action
    # The single output key is "action" with shape [31].  We give each dimension
    # a synthetic name "joint_0 … joint_N" so make_robot_action can build the dict.
    action_key = next(iter(output_features))           # "action"
    action_dim = output_features[action_key]["shape"][0]
    joint_names = [f"joint_{i}" for i in range(action_dim)]
    dataset_features = {
        action_key: {
            "dtype": "float32",
            "shape": output_features[action_key]["shape"],
            "names": joint_names,
        }
    }
    ordered_action_keys = joint_names  # make_robot_action maps tensor[i] → joint_names[i]

    # ------------------------------------------------------------------
    # 2. Load policy
    # ------------------------------------------------------------------
    logger.info("Loading SmolVLA policy weights … (this may take a while)")
    from lerobot.policies.smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()
    logger.info("Policy loaded on device=%s", device)

    # ------------------------------------------------------------------
    # 3. Load preprocessor / postprocessor
    # ------------------------------------------------------------------
    logger.info("Loading preprocessor / postprocessor …")
    preprocessor, postprocessor = _load_processors(args.checkpoint, device)

    # ------------------------------------------------------------------
    # 4. Broker + listener
    # ------------------------------------------------------------------
    from lerobot.rollout.prompt_broker import PromptBroker, StdinPromptListener

    shutdown_event = Event()
    broker = PromptBroker(initial_task=args.task)
    if args.flush_on_switch:
        # Flush the precomputed action chunk immediately so the VLM re-runs with
        # the new task on the very next tick (chunk_size=50 actions discarded).
        broker.register_on_change(policy.flush_action_queue)
    else:
        # Let the current chunk drain naturally; the new task takes effect once
        # the action queue empties and the VLM is called again.
        logger.info("flush_on_switch=off — new task will take effect after current chunk drains")
    StdinPromptListener().start(broker, shutdown_event)

    # ------------------------------------------------------------------
    # 5. Inference engine
    # ------------------------------------------------------------------
    from lerobot.rollout import SyncInferenceConfig, create_inference_engine

    fake_robot_wrapper = MagicMock()
    fake_robot_wrapper.robot_type = "smolvla_demo"

    engine = create_inference_engine(
        SyncInferenceConfig(),
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=fake_robot_wrapper,
        hw_features={},
        dataset_features=dataset_features,
        ordered_action_keys=ordered_action_keys,
        task=args.task,
        fps=args.fps,
        device=device,
        prompt_broker=broker,
    )

    # ------------------------------------------------------------------
    # 6. Pre-build synthetic observation (re-used every tick; copied each call)
    # ------------------------------------------------------------------
    fake_obs_template = _build_fake_obs(input_features)
    logger.info(
        "Synthetic observations: %s",
        {k: (v.shape, v.dtype) for k, v in fake_obs_template.items()},
    )

    # ------------------------------------------------------------------
    # 7. Control loop
    # ------------------------------------------------------------------
    logger.info("=" * 65)
    logger.info("Hot-prompt SmolVLA demo started")
    logger.info("  Checkpoint   : %s", args.checkpoint)
    logger.info("  Initial task : '%s'", args.task)
    logger.info("  FPS          : %.1f", args.fps)
    logger.info("  Device       : %s", device)
    logger.info("  Flush on switch : %s", "yes (immediate VLM re-run)" if args.flush_on_switch else "no (drain current chunk first)")
    logger.info(
        "  Duration     : %s",
        f"{args.duration}s" if args.duration > 0 else "infinite (Ctrl-C to stop)",
    )
    logger.info("=" * 65)
    logger.info("Type a new task below and press Enter to switch the prompt:")

    control_interval = 1.0 / args.fps
    start = time.perf_counter()
    tick = 0

    try:
        while not shutdown_event.is_set():
            loop_start = time.perf_counter()

            if args.duration > 0 and (loop_start - start) >= args.duration:
                logger.info("Duration limit reached (%.0fs)", args.duration)
                break

            # Snapshot the task NOW — this is the exact string the engine will
            # read from the broker a few microseconds later inside get_action().
            # Reading it here (before the call) rather than after avoids a race
            # where a user types a new task during the ~1-2 s inference window,
            # which would make the log show the new task while the policy
            # actually received the old one.
            current_task = broker.get_task()

            # Pass a fresh copy so the engine doesn't mutate the template
            action = engine.get_action({k: v.copy() for k, v in fake_obs_template.items()})
            tick += 1

            # action is a 1-D tensor of shape (action_dim,) after make_robot_action + stack
            preview = action[:5].tolist() if isinstance(action, torch.Tensor) else str(action)[:80]
            logger.info(
                "  tick %4d | task='%s' | action[0:5]=%s",
                tick,
                current_task,
                [f"{v:.4f}" for v in preview],
            )

            elapsed = time.perf_counter() - loop_start
            if (sleep_t := control_interval - elapsed) > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")
    finally:
        shutdown_event.set()

    logger.info("Demo finished after %d ticks", tick)


if __name__ == "__main__":
    main()
