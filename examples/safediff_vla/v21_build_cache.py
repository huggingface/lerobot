"""Build the split + frozen-backbone token cache for the pilot. Usage: uv run python examples/safediff_vla/v21_build_cache.py"""
import json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from v21_common import OUT, build_cache, load_policy, scene_split
rows = pickle.load(open(OUT / "referent_table.pkl", "rb")); split = scene_split(rows); json.dump(split, open(OUT / "pilot_scene_split.json", "w"), indent=1)
import collections; print(collections.Counter((s.split("_")[1], v) for s, v in split.items()))
pol, pre = load_policy(); ent = build_cache(pol, pre, rows, OUT / "pilot_token_cache.pt"); print("cached", len(ent))
