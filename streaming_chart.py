from tqdm import tqdm

from lerobot.common.datasets.streaming_dataset import StreamingLeRobotDataset

repo_id = "lerobot/aloha_mobile_cabinet"
buffer_size = 1000
seed = 42
fps = 50

camera_key = "observation.images.cam_right_wrist"
delta_timestamps = {
    camera_key: [-1, -0.5, -0.20, 0],
    "observation.state": [-1.5, -1, -0.5, -0.20, -0.10, 0],
    "action": [t / fps for t in range(64)],
}

dataset = StreamingLeRobotDataset(
    repo_id=repo_id,
    buffer_size=buffer_size,
    seed=seed,
    delta_timestamps=delta_timestamps,
)
iter_dataset = iter(dataset)

n_samples = 100
for _ in tqdm(range(n_samples)):
    next(iter_dataset)
