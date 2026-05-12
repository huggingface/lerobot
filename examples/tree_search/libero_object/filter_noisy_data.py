from pathlib import Path
import shutil

import numpy as np
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

src_repo_id = "sghosts/noisygain-libero-task0123456789-3pairs_v1"
dst_repo_id = "sghosts/noisygain-libero-task0123456789-3pairs_v2"
dst_root = Path("/content/drive/MyDrive/harezmi-extend-dump/preprocessed_noisygain_v2")

timestamp_min = 2.0

if dst_root.exists():
    shutil.rmtree(dst_root)

print(f"Loading source metadata: {src_repo_id}")
src_meta = LeRobotDatasetMetadata(src_repo_id, force_cache_sync=True)

print(f"Loading source dataset rows: {src_repo_id}")
loaded = load_dataset(src_repo_id)
src_ds = loaded["train"] if isinstance(loaded, DatasetDict) else loaded

print(src_ds)


def image_to_np(value):
    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"))
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict) and value.get("bytes") is not None:
        import io
        return np.asarray(Image.open(io.BytesIO(value["bytes"])).convert("RGB"))
    raise TypeError(f"Unsupported image value type: {type(value)}")


def bool_scalar(value):
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0])
    if isinstance(value, list):
        return bool(value[0])
    return bool(value)


task_by_index = {
    int(row.task_index): str(task)
    for task, row in src_meta.tasks.iterrows()
}

print(f"Creating LeRobot dataset at {dst_root}")
dst = LeRobotDataset.create(
    repo_id=dst_repo_id,
    root=dst_root,
    fps=src_meta.fps,
    features=src_meta.features,
    robot_type=src_meta.robot_type,
    use_videos=False,
    image_writer_threads=4,
)

print(f"Filtering once: timestamp >= {timestamp_min}")
filtered = src_ds.filter(lambda row: float(row["timestamp"]) >= timestamp_min)
print(filtered)

print("Sorting by episode_index then frame_index")
filtered = filtered.sort(["episode_index", "frame_index"])

written = 0
current_episode = None
current_episode_frames = 0

for i, row in enumerate(filtered):
    old_ep = int(row["episode_index"])
    if current_episode is None:
        current_episode = old_ep
    elif old_ep != current_episode:
        dst.save_episode()
        written += 1
        print(
            f"Wrote filtered episode old_ep={current_episode} "
            f"frames={current_episode_frames} new_total={written}"
        )
        current_episode = old_ep
        current_episode_frames = 0

    task_index = int(row["task_index"])
    if task_index not in task_by_index:
        raise KeyError(f"task_index={task_index} is not present in source meta/tasks.parquet")

    dst.add_frame({
        "task": task_by_index[task_index],
        "observation.images.image": image_to_np(row["observation.images.image"]),
        "observation.images.image2": image_to_np(row["observation.images.image2"]),
        "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
        "action": np.asarray(row["action"], dtype=np.float32),
        "is_bad_sequence": np.asarray([bool_scalar(row.get("is_bad_sequence", False))], dtype=np.bool_),
    })
    current_episode_frames += 1

    if (i + 1) % 1000 == 0:
        print(f"Processed rows={i + 1}/{len(filtered)} current_old_ep={current_episode}")

if current_episode is not None and current_episode_frames > 0:
    dst.save_episode()
    written += 1
    print(
        f"Wrote filtered episode old_ep={current_episode} "
        f"frames={current_episode_frames} new_total={written}"
    )

dst.finalize()

print(f"Pushing LeRobot dataset: {dst_repo_id}")
dst.push_to_hub(
    branch="main",
    tag_version=True,
    private=False,
    push_videos=False,
    license="apache-2.0",
)

# Important for image-backed datasets in this repo: push image files too.
images_dir = dst_root / "images"
if images_dir.exists():
    print("Uploading images/ folder")
    HfApi().upload_folder(
        repo_id=dst_repo_id,
        repo_type="dataset",
        folder_path=str(images_dir),
        path_in_repo="images",
    )

print("Done")
