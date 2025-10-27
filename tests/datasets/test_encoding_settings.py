import itertools
import shutil

import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_IMAGES

DUMMY_REPO_ID = "dummy/repo"
DEFAULT_FPS = 30

# I want to test every combination of the following settings:
# the test should create a dataset with one combination, and resume it with another combination.

# [sync, async]
# [1 ep in session, 2 eps in session]


@pytest.mark.parametrize(
    "async_encode, two_eps, resume_async_encode, resume_two_eps",
    list(itertools.product([False, True], repeat=4)),
)
def test_resume_recording(tmp_path, async_encode, two_eps, resume_async_encode, resume_two_eps):
    """
    Tests that:
        - We can record a dataset with a certain combinations of settings
          and resume it with a possibly different combination of settings.
        - Confirms metadata is correctly written
    """
    # tmp_path = Path("tests/artifacts/datasets") / DUMMY_REPO_ID
    shutil.rmtree(tmp_path)
    frames_per_episode = 3

    features = {
        f"{OBS_IMAGES}.cam": {
            "dtype": "video",
            "shape": (80, 40, 3),
            "names": ["height", "width", "channels"],
        },
        ACTION: {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
    }

    video_backend = "pyav"

    # record first session
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID,
        fps=DEFAULT_FPS,
        features=features,
        root=tmp_path,
        use_videos=True,
        video_backend=video_backend,
        async_video_encoding=async_encode,
    )

    episodes_to_record_in_session = 2 if two_eps else 1
    for ep_idx in range(episodes_to_record_in_session):
        for frame_idx in range(frames_per_episode):
            dataset.add_frame(
                {
                    f"{OBS_IMAGES}.cam": torch.zeros((80, 40, 3), dtype=torch.uint8),
                    ACTION: torch.tensor([ep_idx, frame_idx], dtype=torch.float32),
                    "task": f"task_{ep_idx}",
                }
            )
        dataset.save_episode()

    expected_total_episodes = episodes_to_record_in_session

    assert dataset.meta.total_episodes == expected_total_episodes
    assert dataset.meta.total_frames == expected_total_episodes * frames_per_episode

    dataset.finalize()
    initial_root = dataset.root
    initial_repo_id = dataset.repo_id
    del dataset

    # verify correctness of dataset after first session
    dataset_verify = LeRobotDataset(
        initial_repo_id, root=initial_root, revision="v3.0", video_backend=video_backend
    )
    assert dataset_verify.meta.total_episodes == expected_total_episodes
    assert dataset_verify.meta.total_frames == expected_total_episodes * frames_per_episode
    assert len(dataset_verify.hf_dataset) == expected_total_episodes * frames_per_episode

    for idx in range(len(dataset_verify.hf_dataset)):
        item = dataset_verify[idx]
        expected_ep = idx // frames_per_episode
        expected_frame = idx % frames_per_episode
        assert item["episode_index"].item() == expected_ep
        assert item["frame_index"].item() == expected_frame
        assert item["index"].item() == idx
        assert item["action"][0].item() == float(expected_ep)
        assert item["action"][1].item() == float(expected_frame)

    del dataset_verify

    # record second session
    dataset = LeRobotDataset(
        repo_id=DUMMY_REPO_ID,
        root=tmp_path,
        video_backend=video_backend,
        async_video_encoding=resume_async_encode,
    )

    episodes_to_record_in_session = 2 if resume_two_eps else 1
    for ep_idx in range(expected_total_episodes, expected_total_episodes + episodes_to_record_in_session):
        for frame_idx in range(frames_per_episode):
            dataset.add_frame(
                {
                    f"{OBS_IMAGES}.cam": torch.zeros((80, 40, 3), dtype=torch.uint8),
                    ACTION: torch.tensor([ep_idx, frame_idx], dtype=torch.float32),
                    "task": f"task_{ep_idx}",
                }
            )
        dataset.save_episode()

    expected_total_episodes += episodes_to_record_in_session

    assert dataset.meta.total_episodes == expected_total_episodes
    assert dataset.meta.total_frames == expected_total_episodes * frames_per_episode

    dataset.finalize()
    initial_root = dataset.root
    initial_repo_id = dataset.repo_id
    del dataset
    import time

    time.sleep(1)

    # verify correctness of dataset after second session
    dataset_verify = LeRobotDataset(
        initial_repo_id, root=initial_root, revision="v3.0", video_backend=video_backend
    )
    assert dataset_verify.meta.total_episodes == expected_total_episodes
    assert dataset_verify.meta.total_frames == expected_total_episodes * frames_per_episode
    assert len(dataset_verify.hf_dataset) == expected_total_episodes * frames_per_episode

    for idx in range(len(dataset_verify.hf_dataset)):
        item = dataset_verify[idx]
        expected_ep = idx // frames_per_episode
        expected_frame = idx % frames_per_episode
        assert item["episode_index"].item() == expected_ep
        assert item["frame_index"].item() == expected_frame
        assert item["index"].item() == idx
        assert item["action"][0].item() == float(expected_ep)
        assert item["action"][1].item() == float(expected_frame)

    del dataset_verify
