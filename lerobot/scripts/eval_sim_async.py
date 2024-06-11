import argparse
import logging
import threading
import time
from collections import deque
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path
from threading import Lock

import cv2
import gymnasium as gym
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils._errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.logger import log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed


def rollout(env: gym.vector.VectorEnv, policy: Policy, seed: int | None = None):
    device = get_device_from_parameters(policy)

    fps = env.unwrapped.metadata["render_fps"]
    period = 1 / fps

    window_name = "window"

    # warmup
    policy.reset()
    observation, _ = env.reset(seed=seed)
    cv2.imshow(window_name, observation["pixels"]["top"][0])
    observation = preprocess_observation(observation)
    observation = {key: observation[key].to(device, non_blocking=True) for key in observation}
    for _ in range(policy.config.n_action_steps * 3):
        with torch.inference_mode():
            action = policy.select_action(observation)

    policy.reset()
    observation, _ = env.reset(seed=seed)
    actions_queue = deque()
    action = observation["agent_pos"].copy()
    actions_queue.append(observation["agent_pos"].copy())
    future_observation, _, terminated, truncated, _ = env.step(actions_queue.pop())
    done = terminated ^ truncated
    drop_action_count = 1

    lock = Lock()

    def run_policy():
        start = time.time()
        nonlocal actions_queue, drop_action_count
        inp = preprocess_observation(observation)
        inp = {key: inp[key].to(device, non_blocking=True) for key in inp}
        with torch.inference_mode():
            actions = policy.select_action(inp)
        # Simulate extra infernce time.
        time.sleep(0.2)
        lock.acquire()
        actions_queue.clear()
        actions_queue.extend(actions.transpose(0, 1).cpu().numpy()[drop_action_count:])
        lock.release()
        drop_action_count = 1
        print(f"Policy time: {time.time() - start}")

    # For the first step, we can take our time to run the policy.
    thread = threading.Thread(target=run_policy)
    thread.start()
    thread.join()

    while not done:
        cv2.imshow("window", cv2.cvtColor(observation["pixels"]["top"][0], cv2.COLOR_BGR2RGB))
        # Pretend this is when we get the observation.
        start = time.time()
        # If we have less than some number of actions left in the queue, we need to start working on producing
        # the next chunk.
        if len(actions_queue) < 2 and not thread.is_alive():
            thread = threading.Thread(target=run_policy)
            thread.start()
        # # Process the observation that we have right now to decide an action.
        # inp = preprocess_observation(observation)
        # inp = {key: inp[key].to(device, non_blocking=True) for key in inp}
        # with torch.inference_mode():
        #     action = policy.select_action(inp)
        # action = action.to("cpu").numpy()
        # Simulate a clock cycle (break out early if we have already finished running the policy).
        while time.time() - start < period:
            time.sleep(0.001)
        # Try to get an action from the queue.
        lock.acquire()
        if len(actions_queue) == 0:
            print("DROPPED CYCLE!")
            # We've dropped a cycle anyway, so let the policy catch up so we can continue.
            lock.release()
            thread.join()
            lock.acquire()
        action = actions_queue.popleft()
        # If the thread is working we need to make sure that it drops another action from whatever it ends
        # up producing
        if thread.is_alive():
            drop_action_count += 1
        lock.release()
        # In the next loop iteration we'll have access to what is currently the future observation.
        observation = deepcopy(future_observation)
        # Send the action to the env, but it will only be executed on the next clock cycle.
        # `future_observation`is what we should see one clock cycle after the action is executed.
        future_observation, _, terminated, truncated, _ = env.step(action)
        done = terminated ^ truncated
        cv2.waitKey(max(1, int(round((period - (time.time() - start)) * 1000))))


def main(
    pretrained_policy_path: str | None = None,
    hydra_cfg_path: str | None = None,
    out_dir: str | None = None,
    config_overrides: list[str] | None = None,
):
    assert (pretrained_policy_path is None) ^ (hydra_cfg_path is None)
    if hydra_cfg_path is None:
        hydra_cfg = init_hydra_config(pretrained_policy_path / "config.yaml", config_overrides)
    else:
        hydra_cfg = init_hydra_config(hydra_cfg_path, config_overrides)
    if out_dir is None:
        out_dir = f"outputs/eval/{dt.now().strftime('%Y-%m-%d/%H-%M-%S')}_{hydra_cfg.env.name}_{hydra_cfg.policy.name}"

    hydra_cfg.eval.n_episodes = 1
    hydra_cfg.eval.batch_size = 1

    # Check device is available
    device = get_safe_torch_device(hydra_cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)

    log_output_dir(out_dir)

    logging.info("Making environment.")
    env = make_env(hydra_cfg)

    logging.info("Making policy.")
    if hydra_cfg_path is None:
        policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=pretrained_policy_path)
    else:
        # Note: We need the dataset stats to pass to the policy's normalization modules.
        policy = make_policy(hydra_cfg=hydra_cfg, dataset_stats=make_dataset(hydra_cfg).stats)
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if hydra_cfg.use_amp else nullcontext():
        rollout(env, policy, seed=hydra_cfg.seed)


if __name__ == "__main__":
    init_logging()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch "
            "(useful for debugging). This argument is mutually exclusive with `--config`."
        ),
    )
    group.add_argument(
        "--config",
        help=(
            "Path to a yaml config you want to use for initializing a policy from scratch (useful for "
            "debugging). This argument is mutually exclusive with `--pretrained-policy-name-or-path` (`-p`)."
        ),
    )
    parser.add_argument("--revision", help="Optionally provide the Hugging Face Hub revision ID.")
    parser.add_argument(
        "--out-dir",
        help=(
            "Where to save the evaluation outputs. If not provided, outputs are saved in "
            "outputs/eval/{timestamp}_{env_name}_{policy_name}"
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()

    try:
        pretrained_policy_path = Path(
            snapshot_download(args.pretrained_policy_name_or_path, revision=args.revision)
        )
    except (HFValidationError, RepositoryNotFoundError) as e:
        if isinstance(e, HFValidationError):
            error_message = (
                "The provided pretrained_policy_name_or_path is not a valid Hugging Face Hub repo ID."
            )
        else:
            error_message = (
                "The provided pretrained_policy_name_or_path was not found on the Hugging Face Hub."
            )

        logging.warning(f"{error_message} Treating it as a local directory.")
        pretrained_policy_path = Path(args.pretrained_policy_name_or_path)
    if not pretrained_policy_path.is_dir() or not pretrained_policy_path.exists():
        raise ValueError(
            "The provided pretrained_policy_name_or_path is not a valid/existing Hugging Face Hub "
            "repo ID, nor is it an existing local directory."
        )

    main(
        pretrained_policy_path=pretrained_policy_path,
        out_dir=args.out_dir,
        config_overrides=args.overrides,
    )
