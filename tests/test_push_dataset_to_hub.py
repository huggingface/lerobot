"""
This file contains generic tests to ensure that nothing breaks if we modify the push_dataset_to_hub API.
Also, this file contains backward compatibility tests. Because they are slow and require to download the raw datasets,
we skip them for now in our CI.

Example to run backward compatiblity tests locally:
```
DATA_DIR=tests/data python -m pytest --run-skipped tests/test_push_dataset_to_hub.py::test_push_dataset_to_hub_pusht_backward_compatibility
```
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import save_images_concurrently
from lerobot.common.datasets.video_utils import encode_video_frames
from lerobot.scripts.push_dataset_to_hub import push_dataset_to_hub
from tests.utils import require_package_arg


def _mock_download_raw_pusht(raw_dir, num_frames=4, num_episodes=3):
    import zarr

    raw_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = raw_dir / "pusht_cchi_v7_replay.zarr"
    store = zarr.DirectoryStore(zarr_path)
    zarr_data = zarr.group(store=store)

    zarr_data.create_dataset(
        "data/action", shape=(num_frames, 1), chunks=(num_frames, 1), dtype=np.float32, overwrite=True
    )
    zarr_data.create_dataset(
        "data/img",
        shape=(num_frames, 96, 96, 3),
        chunks=(num_frames, 96, 96, 3),
        dtype=np.uint8,
        overwrite=True,
    )
    zarr_data.create_dataset(
        "data/n_contacts", shape=(num_frames, 2), chunks=(num_frames, 2), dtype=np.float32, overwrite=True
    )
    zarr_data.create_dataset(
        "data/state", shape=(num_frames, 5), chunks=(num_frames, 5), dtype=np.float32, overwrite=True
    )
    zarr_data.create_dataset(
        "data/keypoint", shape=(num_frames, 9, 2), chunks=(num_frames, 9, 2), dtype=np.float32, overwrite=True
    )
    zarr_data.create_dataset(
        "meta/episode_ends", shape=(num_episodes,), chunks=(num_episodes,), dtype=np.int32, overwrite=True
    )

    zarr_data["data/action"][:] = np.random.randn(num_frames, 1)
    zarr_data["data/img"][:] = np.random.randint(0, 255, size=(num_frames, 96, 96, 3), dtype=np.uint8)
    zarr_data["data/n_contacts"][:] = np.random.randn(num_frames, 2)
    zarr_data["data/state"][:] = np.random.randn(num_frames, 5)
    zarr_data["data/keypoint"][:] = np.random.randn(num_frames, 9, 2)
    zarr_data["meta/episode_ends"][:] = np.array([1, 3, 4])

    store.close()


def _mock_download_raw_umi(raw_dir, num_frames=4, num_episodes=3):
    import zarr

    raw_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = raw_dir / "cup_in_the_wild.zarr"
    store = zarr.DirectoryStore(zarr_path)
    zarr_data = zarr.group(store=store)

    zarr_data.create_dataset(
        "data/camera0_rgb",
        shape=(num_frames, 96, 96, 3),
        chunks=(num_frames, 96, 96, 3),
        dtype=np.uint8,
        overwrite=True,
    )
    zarr_data.create_dataset(
        "data/robot0_demo_end_pose",
        shape=(num_frames, 5),
        chunks=(num_frames, 5),
        dtype=np.float32,
        overwrite=True,
    )
    zarr_data.create_dataset(
        "data/robot0_demo_start_pose",
        shape=(num_frames, 5),
        chunks=(num_frames, 5),
        dtype=np.float32,
        overwrite=True,
    )
    zarr_data.create_dataset(
        "data/robot0_eef_pos", shape=(num_frames, 5), chunks=(num_frames, 5), dtype=np.float32, overwrite=True
    )
    zarr_data.create_dataset(
        "data/robot0_eef_rot_axis_angle",
        shape=(num_frames, 5),
        chunks=(num_frames, 5),
        dtype=np.float32,
        overwrite=True,
    )
    zarr_data.create_dataset(
        "data/robot0_gripper_width",
        shape=(num_frames, 5),
        chunks=(num_frames, 5),
        dtype=np.float32,
        overwrite=True,
    )
    zarr_data.create_dataset(
        "meta/episode_ends", shape=(num_episodes,), chunks=(num_episodes,), dtype=np.int32, overwrite=True
    )

    zarr_data["data/camera0_rgb"][:] = np.random.randint(0, 255, size=(num_frames, 96, 96, 3), dtype=np.uint8)
    zarr_data["data/robot0_demo_end_pose"][:] = np.random.randn(num_frames, 5)
    zarr_data["data/robot0_demo_start_pose"][:] = np.random.randn(num_frames, 5)
    zarr_data["data/robot0_eef_pos"][:] = np.random.randn(num_frames, 5)
    zarr_data["data/robot0_eef_rot_axis_angle"][:] = np.random.randn(num_frames, 5)
    zarr_data["data/robot0_gripper_width"][:] = np.random.randn(num_frames, 5)
    zarr_data["meta/episode_ends"][:] = np.array([1, 3, 4])

    store.close()


def _mock_download_raw_xarm(raw_dir, num_frames=4):
    import pickle

    dataset_dict = {
        "observations": {
            "rgb": np.random.randint(0, 255, size=(num_frames, 3, 84, 84), dtype=np.uint8),
            "state": np.random.randn(num_frames, 4),
        },
        "actions": np.random.randn(num_frames, 3),
        "rewards": np.random.randn(num_frames),
        "masks": np.random.randn(num_frames),
        "dones": np.array([False, True, True, True]),
    }

    raw_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = raw_dir / "buffer.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(dataset_dict, f)


def _mock_download_raw_aloha(raw_dir, num_frames=6, num_episodes=3):
    import h5py

    for ep_idx in range(num_episodes):
        raw_dir.mkdir(parents=True, exist_ok=True)
        path_h5 = raw_dir / f"episode_{ep_idx}.hdf5"
        with h5py.File(str(path_h5), "w") as f:
            f.create_dataset("action", data=np.random.randn(num_frames // num_episodes, 14))
            f.create_dataset("observations/qpos", data=np.random.randn(num_frames // num_episodes, 14))
            f.create_dataset("observations/qvel", data=np.random.randn(num_frames // num_episodes, 14))
            f.create_dataset(
                "observations/images/top",
                data=np.random.randint(
                    0, 255, size=(num_frames // num_episodes, 480, 640, 3), dtype=np.uint8
                ),
            )


def _mock_download_raw_dora(raw_dir, num_frames=6, num_episodes=3, fps=30):
    from datetime import datetime, timedelta, timezone

    import pandas

    def write_parquet(key, timestamps, values):
        data = {
            "timestamp_utc": timestamps,
            key: values,
        }
        df = pandas.DataFrame(data)
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(raw_dir / f"{key}.parquet", engine="pyarrow")

    episode_indices = [None, None, -1, None, None, -1, None, None, -1]
    episode_indices_mapping = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    frame_indices = [0, 1, -1, 0, 1, -1, 0, 1, -1]

    cam_key = "observation.images.cam_high"
    timestamps = []
    actions = []
    states = []
    frames = []
    # `+ num_episodes`` for buffer frames associated to episode_index=-1
    for i, frame_idx in enumerate(frame_indices):
        t_utc = datetime.now(timezone.utc) + timedelta(seconds=i / fps)
        action = np.random.randn(21).tolist()
        state = np.random.randn(21).tolist()
        ep_idx = episode_indices_mapping[i]
        frame = [{"path": f"videos/{cam_key}_episode_{ep_idx:06d}.mp4", "timestamp": frame_idx / fps}]
        timestamps.append(t_utc)
        actions.append(action)
        states.append(state)
        frames.append(frame)

    write_parquet(cam_key, timestamps, frames)
    write_parquet("observation.state", timestamps, states)
    write_parquet("action", timestamps, actions)
    write_parquet("episode_index", timestamps, episode_indices)

    # write fake mp4 file for each episode
    for ep_idx in range(num_episodes):
        imgs_array = np.random.randint(0, 255, size=(num_frames // num_episodes, 480, 640, 3), dtype=np.uint8)

        tmp_imgs_dir = raw_dir / "tmp_images"
        save_images_concurrently(imgs_array, tmp_imgs_dir)

        fname = f"{cam_key}_episode_{ep_idx:06d}.mp4"
        video_path = raw_dir / "videos" / fname
        encode_video_frames(tmp_imgs_dir, video_path, fps, vcodec="libx264")


def _mock_download_raw(raw_dir, repo_id):
    if "wrist_gripper" in repo_id:
        _mock_download_raw_dora(raw_dir)
    elif "aloha" in repo_id:
        _mock_download_raw_aloha(raw_dir)
    elif "pusht" in repo_id:
        _mock_download_raw_pusht(raw_dir)
    elif "xarm" in repo_id:
        _mock_download_raw_xarm(raw_dir)
    elif "umi" in repo_id:
        _mock_download_raw_umi(raw_dir)
    else:
        raise ValueError(repo_id)


def test_push_dataset_to_hub_invalid_repo_id(tmpdir):
    with pytest.raises(ValueError):
        push_dataset_to_hub(Path(tmpdir), "raw_format", "invalid_repo_id")


def test_push_dataset_to_hub_out_dir_force_override_false(tmpdir):
    tmpdir = Path(tmpdir)
    out_dir = tmpdir / "out"
    raw_dir = tmpdir / "raw"
    # mkdir to skip download
    raw_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        push_dataset_to_hub(
            raw_dir=raw_dir,
            raw_format="some_format",
            repo_id="user/dataset",
            local_dir=out_dir,
            force_override=False,
        )


@pytest.mark.parametrize(
    "required_packages, raw_format, repo_id, make_test_data",
    [
        (["gym_pusht"], "pusht_zarr", "lerobot/pusht", False),
        (["gym_pusht"], "pusht_zarr", "lerobot/pusht", True),
        (None, "xarm_pkl", "lerobot/xarm_lift_medium", False),
        (None, "aloha_hdf5", "lerobot/aloha_sim_insertion_scripted", False),
        (["imagecodecs"], "umi_zarr", "lerobot/umi_cup_in_the_wild", False),
        (None, "dora_parquet", "cadene/wrist_gripper", False),
    ],
)
@require_package_arg
def test_push_dataset_to_hub_format(required_packages, tmpdir, raw_format, repo_id, make_test_data):
    num_episodes = 3
    tmpdir = Path(tmpdir)

    raw_dir = tmpdir / f"{repo_id}_raw"
    _mock_download_raw(raw_dir, repo_id)

    local_dir = tmpdir / repo_id

    lerobot_dataset = push_dataset_to_hub(
        raw_dir=raw_dir,
        raw_format=raw_format,
        repo_id=repo_id,
        push_to_hub=False,
        local_dir=local_dir,
        force_override=False,
        cache_dir=tmpdir / "cache",
        tests_data_dir=tmpdir / "tests/data" if make_test_data else None,
        encoding={"vcodec": "libx264"},
    )

    # minimal generic tests on the local directory containing LeRobotDataset
    assert (local_dir / "meta_data" / "info.json").exists()
    assert (local_dir / "meta_data" / "stats.safetensors").exists()
    assert (local_dir / "meta_data" / "episode_data_index.safetensors").exists()
    for i in range(num_episodes):
        for cam_key in lerobot_dataset.camera_keys:
            assert (local_dir / "videos" / f"{cam_key}_episode_{i:06d}.mp4").exists()
    assert (local_dir / "train" / "dataset_info.json").exists()
    assert (local_dir / "train" / "state.json").exists()
    assert len(list((local_dir / "train").glob("*.arrow"))) > 0

    # minimal generic tests on the item
    item = lerobot_dataset[0]
    assert "index" in item
    assert "episode_index" in item
    assert "timestamp" in item
    for cam_key in lerobot_dataset.camera_keys:
        assert cam_key in item

    if make_test_data:
        # Check that only the first episode is selected.
        test_dataset = LeRobotDataset(repo_id=repo_id, root=tmpdir / "tests/data")
        num_frames = sum(
            i == lerobot_dataset.hf_dataset["episode_index"][0]
            for i in lerobot_dataset.hf_dataset["episode_index"]
        ).item()
        assert (
            test_dataset.hf_dataset["episode_index"]
            == lerobot_dataset.hf_dataset["episode_index"][:num_frames]
        )
        for k in ["from", "to"]:
            assert torch.equal(test_dataset.episode_data_index[k], lerobot_dataset.episode_data_index[k][:1])


@pytest.mark.parametrize(
    "raw_format, repo_id",
    [
        # TODO(rcadene): add raw dataset test artifacts
        ("pusht_zarr", "lerobot/pusht"),
        ("xarm_pkl", "lerobot/xarm_lift_medium"),
        ("aloha_hdf5", "lerobot/aloha_sim_insertion_scripted"),
        ("umi_zarr", "lerobot/umi_cup_in_the_wild"),
        ("dora_parquet", "cadene/wrist_gripper"),
    ],
)
@pytest.mark.skip(
    "Not compatible with our CI since it downloads raw datasets. Run with `DATA_DIR=tests/data python -m pytest --run-skipped tests/test_push_dataset_to_hub.py::test_push_dataset_to_hub_pusht_backward_compatibility`"
)
def test_push_dataset_to_hub_pusht_backward_compatibility(tmpdir, raw_format, repo_id):
    _, dataset_id = repo_id.split("/")

    tmpdir = Path(tmpdir)
    raw_dir = tmpdir / f"{dataset_id}_raw"
    local_dir = tmpdir / repo_id

    push_dataset_to_hub(
        raw_dir=raw_dir,
        raw_format=raw_format,
        repo_id=repo_id,
        push_to_hub=False,
        local_dir=local_dir,
        force_override=False,
        cache_dir=tmpdir / "cache",
        episodes=[0],
    )

    ds_actual = LeRobotDataset(repo_id, root=tmpdir)
    ds_reference = LeRobotDataset(repo_id)

    assert len(ds_reference.hf_dataset) == len(ds_actual.hf_dataset)

    def check_same_items(item1, item2):
        assert item1.keys() == item2.keys(), "Keys mismatch"

        for key in item1:
            if isinstance(item1[key], torch.Tensor) and isinstance(item2[key], torch.Tensor):
                assert torch.equal(item1[key], item2[key]), f"Mismatch found in key: {key}"
            else:
                assert item1[key] == item2[key], f"Mismatch found in key: {key}"

    for i in range(len(ds_reference.hf_dataset)):
        item_reference = ds_reference.hf_dataset[i]
        item_actual = ds_actual.hf_dataset[i]
        check_same_items(item_reference, item_actual)
