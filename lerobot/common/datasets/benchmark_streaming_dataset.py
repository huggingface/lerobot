import time

import pandas as pd
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.common.datasets.utils import get_hf_dataset_size_in_mb

repo_id = "lerobot/aloha_mobile_cabinet"

camera_key = "observation.images.cam_right_wrist"
fps = 50

delta_timestamps = {
    camera_key: [-1, -0.5, -0.20, 0],
    "observation.state": [-1.5, -1, -0.5, -0.20, -0.10, 0],
    "action": [t / fps for t in range(6)],
}

regular_dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
streaming_dataset = StreamingLeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

iter_streaming_dataset = iter(streaming_dataset)

print(f"All-in-memory size: {get_hf_dataset_size_in_mb(regular_dataset.hf_dataset)}")

n_trials = 1_000
results = []

for _i in tqdm(range(n_trials), desc="Benchmarking dataset iteration"):
    start_time = time.perf_counter()
    next(iter_streaming_dataset)
    streaming_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    _ = regular_dataset[_i]
    regular_time = time.perf_counter() - start_time

    results.append({"next_time": streaming_time, "is_streaming": True})
    results.append({"next_time": regular_time, "is_streaming": False})

df = pd.DataFrame(results)
df.to_csv("benchmark_streaming_dataset.csv", index=False)

print(df.groupby("is_streaming").next_time.agg(["mean", "std", "count"]))
