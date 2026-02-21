"""Pure unit tests for rename logic that do not import the full package.

These tests avoid heavy imports so they can run in minimal CI/dev environments.
"""

def rename_stats(stats: dict, rename_map: dict) -> dict:
    from copy import deepcopy

    if not stats:
        return {}
    renamed = {}
    for old_key, sub_stats in stats.items():
        new_key = rename_map.get(old_key, old_key)
        renamed[new_key] = deepcopy(sub_stats) if sub_stats is not None else {}
    return renamed


def test_rename_stats_top_level_keys():
    stats = {
        "observation.images.front": {"mean": 1},
        "observation.state": {"mean": 0},
    }
    rename_map = {"observation.images.front": "observation.images.camera1"}

    renamed = rename_stats(stats, rename_map)

    assert "observation.images.camera1" in renamed
    assert "observation.images.front" not in renamed
    assert renamed["observation.images.camera1"]["mean"] == 1


def test_obs_key_rename_logic_matches_record_loop():
    obs_processed = {"front": "img_front", "side": "img_side", "something": 42}

    rename_map = {
        "observation.images.front": "observation.images.camera1",
        "observation.images.side": "observation.images.camera2",
    }

    image_map = {}
    pref = "observation.images."
    for old_full, new_full in rename_map.items():
        if old_full.startswith(pref) and new_full.startswith(pref):
            old_suffix = old_full[len(pref) :]
            new_suffix = new_full[len(pref) :]
            image_map[old_suffix] = new_suffix

    obs_for_build = dict(obs_processed)
    for old_suffix, new_suffix in image_map.items():
        if old_suffix in obs_for_build and new_suffix not in obs_for_build:
            obs_for_build[new_suffix] = obs_for_build.pop(old_suffix)

    assert "camera1" in obs_for_build and "camera2" in obs_for_build
    assert "front" not in obs_for_build and "side" not in obs_for_build
    assert obs_for_build["camera1"] == "img_front"
    assert obs_for_build["camera2"] == "img_side"
