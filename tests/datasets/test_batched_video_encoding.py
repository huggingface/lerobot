import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import get_video_duration_in_s
from tests.fixtures.constants import DUMMY_REPO_ID, DUMMY_ROBOT_TYPE


def test_batched_video_encoding_advances_timestamps(tmp_path, features_factory):
    """
    Regression test for batched video encoding:

    When `batch_encoding_size > 1`, videos are encoded after multiple episodes are saved. The dataset stores
    one mp4 per (video_key, chunk, file) and relies on per-episode `from_timestamp`/`to_timestamp` to map each
    episode to its segment in that mp4. If those timestamps don't advance, every episode will decode the same
    video segment.
    """
    root = tmp_path / "batched_video_ds"
    repo_id = f"{DUMMY_REPO_ID}_batched_video"
    fps = 10
    features = features_factory(use_videos=True)

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        robot_type=DUMMY_ROBOT_TYPE,
        root=root,
        use_videos=True,
        batch_encoding_size=2,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    # Episode 0: black frames
    black = np.zeros((64, 96, 3), dtype=np.uint8)
    for _ in range(5):
        ds.add_frame(
            {
                "action": np.zeros((6,), dtype=np.float32),
                "state": np.zeros((6,), dtype=np.float32),
                "laptop": black,
                "phone": black,
                "task": "dummy_task",
            }
        )
    ds.save_episode(parallel_encoding=False)

    # Episode 1: white frames
    white = np.full((64, 96, 3), 255, dtype=np.uint8)
    for _ in range(5):
        ds.add_frame(
            {
                "action": np.zeros((6,), dtype=np.float32),
                "state": np.zeros((6,), dtype=np.float32),
                "laptop": white,
                "phone": white,
                "task": "dummy_task",
            }
        )
    ds.save_episode(parallel_encoding=False)

    # After 2 episodes, batch encoding should have run. Timestamps should advance per episode.
    for vid_key in ds.meta.video_keys:
        ep0 = ds.meta.episodes[0]
        ep1 = ds.meta.episodes[1]
        assert ep0[f"videos/{vid_key}/from_timestamp"] == 0.0
        assert ep1[f"videos/{vid_key}/from_timestamp"] > 0.0
        assert ep1[f"videos/{vid_key}/to_timestamp"] > ep0[f"videos/{vid_key}/to_timestamp"]

        # The mp4 should contain both episodes (duration > single-episode duration).
        video_path = ds.root / ds.meta.get_video_file_path(1, vid_key)
        assert video_path.exists()
        duration_s = get_video_duration_in_s(video_path)
        assert duration_s > ep0[f"videos/{vid_key}/to_timestamp"]
