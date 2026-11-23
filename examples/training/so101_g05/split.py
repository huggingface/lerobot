"""Use the same deterministic per-family holdout as the FineART pilot."""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def held_out(root: Path):
    groups = defaultdict(list)
    for line in (root / "meta/generation/episodes.jsonl").read_text().splitlines():
        row = json.loads(line)
        groups[row["task"]].append(row["episode_index"])
    rng = random.Random(1000)
    result = []
    for values in groups.values():
        rng.shuffle(values)
        result.extend(values[: math.ceil(len(values) * 0.05)])
    return sorted(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(held_out(args.root)))
