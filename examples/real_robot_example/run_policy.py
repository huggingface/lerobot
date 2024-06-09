import argparse
import logging
from pathlib import Path

import gym_real_world  # noqa: F401
import gymnasium as gym  # noqa: F401
from huggingface_hub import snapshot_download
from huggingface_hub.utils._errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError

from lerobot.common.utils.utils import init_logging
from lerobot.scripts.eval import eval

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
    parser.add_argument("--revision", help="Optionally provide the Hugging Face Hub revision ID.")
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

    eval(pretrained_policy_path=pretrained_policy_path, config_overrides=args.overrides)
