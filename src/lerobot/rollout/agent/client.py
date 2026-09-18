# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Call the local LeRobot agent API from Codex, Claude, or another tool-using agent."""

import argparse
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8767")
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--arguments", default="{}")
    parser.add_argument("--revision", type=int)
    parser.add_argument("--observed-at", type=float)
    args = parser.parse_args()
    result = requests.post(
        args.url + "/call",
        timeout=60,
        headers={"Authorization": "Bearer " + Path(args.token_file).read_text().strip()},
        json={
            "name": args.name,
            "arguments": json.loads(args.arguments),
            "revision": args.revision,
            "observed_at": args.observed_at,
        },
    )
    print(json.dumps(result.json(), indent=2))
    result.raise_for_status()


if __name__ == "__main__":
    main()
