"""Simple CLI to print registered action contracts and compatibility checks."""
from __future__ import annotations

from lerobot.action_semantics import (
    LIBERO_ACTION_CONTRACT,
    LIBERO_DATASET_ACTION_CONTRACT,
    LIBERO_ENV_ACTION_CONTRACT,
    LIBERO_SAFETY_DATASET_ACTION_CONTRACT,
    LIBERO_SAFETY_ENV_ACTION_CONTRACT,
    convert_action,
)

def _describe(c):
    return {
        "name": c.name,
        "fps": c.fps,
        "representation": c.representation,
        "controller_translation_scale": c.controller_translation_scale,
        "controller_rotation_scale": c.controller_rotation_scale,
    }


def main():
    print("=== Contracts ===")
    for c in [
        LIBERO_ACTION_CONTRACT,
        LIBERO_DATASET_ACTION_CONTRACT,
        LIBERO_ENV_ACTION_CONTRACT,
        LIBERO_SAFETY_DATASET_ACTION_CONTRACT,
        LIBERO_SAFETY_ENV_ACTION_CONTRACT,
    ]:
        print(_describe(c))


if __name__ == "__main__":
    main()
