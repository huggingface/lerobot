"""Small standalone smoke test to verify rename logic without importing the full package.

This avoids heavy imports like `torch` or `serial` so it can be executed in minimal environments.
"""

from copy import deepcopy


def rename_stats(stats: dict, rename_map: dict) -> dict:
    if not stats:
        return {}
    renamed = {}
    for old_key, sub_stats in stats.items():
        new_key = rename_map.get(old_key, old_key)
        renamed[new_key] = deepcopy(sub_stats) if sub_stats is not None else {}
    return renamed


def apply_obs_rename(obs_processed: dict, rename_map: dict, obs_prefix: str = "observation") -> dict:
    obs = dict(obs_processed)
    pref = f"{obs_prefix}.images."
    image_map = {}
    for old_full, new_full in rename_map.items():
        if old_full.startswith(pref) and new_full.startswith(pref):
            old_suffix = old_full.removeprefix(pref)
            new_suffix = new_full.removeprefix(pref)
            image_map[old_suffix] = new_suffix

    for old_suffix, new_suffix in image_map.items():
        if old_suffix in obs and new_suffix not in obs:
            obs[new_suffix] = obs.pop(old_suffix)

    return obs


def main() -> int:
    # Test rename_stats
    stats = {
        "observation.images.front": {"mean": 1},
        "observation.state": {"mean": 0},
    }
    rename_map = {"observation.images.front": "observation.images.camera1"}
    renamed = rename_stats(stats, rename_map)
    assert "observation.images.camera1" in renamed and "observation.images.front" not in renamed

    # Test obs rename
    obs_processed = {"front": "img_front", "side": "img_side", "other": 7}
    rename_map = {
        "observation.images.front": "observation.images.camera1",
        "observation.images.side": "observation.images.camera2",
    }
    obs_renamed = apply_obs_rename(obs_processed, rename_map)
    assert "camera1" in obs_renamed and "camera2" in obs_renamed
    assert "front" not in obs_renamed and "side" not in obs_renamed

    print("Smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
