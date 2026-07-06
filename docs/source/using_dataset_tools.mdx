# Using Dataset Tools

This guide covers the dataset tools utilities available in LeRobot for modifying and editing existing datasets.

## Overview

LeRobot provides several utilities for manipulating datasets:

1. **Delete Episodes** - Remove specific episodes from a dataset
2. **Split Dataset** - Divide a dataset into multiple smaller datasets
3. **Merge Datasets** - Combine multiple datasets into one. The datasets must have identical features, and episodes are concatenated in the order specified in `repo_ids`
4. **Add Features** - Add new features to a dataset
5. **Remove Features** - Remove features from a dataset
6. **Convert to Video** - Convert image-based datasets to video format for efficient storage (RGB and depth cameras are encoded with separate encoders)
7. **Re-encode Videos** - Re-encode an existing video dataset's RGB and/or depth streams with new encoder settings
8. **Show the Info of Datasets** - Show the summary of datasets information such as number of episode etc.

The core implementation is in `lerobot.datasets.dataset_tools`.
An example script detailing how to use the tools API is available in `examples/dataset/use_dataset_tools.py`.

## Command-Line Tool: lerobot-edit-dataset

`lerobot-edit-dataset` is a command-line script for editing datasets. It can be used to delete episodes, split datasets, merge datasets, add features, remove features, and convert image datasets to video format.

Run `lerobot-edit-dataset --help` for more information on the configuration of each operation.

### Usage Examples

#### Delete Episodes

Remove specific episodes from a dataset. This is useful for filtering out undesired data.

```bash
# Delete episodes 0, 2, and 5 (modifies original dataset)
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --operation.type delete_episodes \
    --operation.episode_indices "[0, 2, 5]"

# Delete episodes and save to a new dataset (preserves original dataset)
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --new_repo_id lerobot/pusht_after_deletion \
    --operation.type delete_episodes \
    --operation.episode_indices "[0, 2, 5]"
```

#### Split Dataset

Divide a dataset into multiple subsets.

```bash
# Split by fractions (e.g. 80% train, 20% test, 20% val)
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --operation.type split \
    --operation.splits '{"train": 0.8, "test": 0.2, "val": 0.2}'

# Split by specific episode indices
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --operation.type split \
    --operation.splits '{"task1": [0, 1, 2, 3], "task2": [4, 5]}'
```

There are no constraints on the split names, they can be determined by the user. Resulting datasets are saved under the repo id with the split name appended, e.g. `lerobot/pusht_train`, `lerobot/pusht_task1`, `lerobot/pusht_task2`.

#### Merge Datasets

Combine multiple datasets into a single dataset.

```bash
# Merge train and validation splits back into one dataset
lerobot-edit-dataset \
    --repo_id lerobot/pusht_merged \
    --operation.type merge \
    --operation.repo_ids "['lerobot/pusht_train', 'lerobot/pusht_val']"
```

#### Remove Features

Remove features from a dataset.

```bash
# Remove a camera feature
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --operation.type remove_feature \
    --operation.feature_names "['observation.images.top']"
```

#### Convert to Video

Convert an image-based dataset to video format, creating a new LeRobotDataset where images are stored as videos. This is useful for reducing storage requirements and improving data loading performance. The new dataset will have the exact same structure as the original, but with images encoded as MP4 videos in the proper LeRobot format.

```bash
# Local-only: Save to a custom output directory (no hub push)
lerobot-edit-dataset \
    --repo_id lerobot/pusht_image \
    --operation.type convert_image_to_video \
    --operation.output_dir /path/to/output/pusht_video

# Save with new repo_id (local storage)
lerobot-edit-dataset \
    --repo_id lerobot/pusht_image \
    --new_repo_id lerobot/pusht_video \
    --operation.type convert_image_to_video

# Convert and push to Hugging Face Hub
lerobot-edit-dataset \
    --repo_id lerobot/pusht_image \
    --new_repo_id lerobot/pusht_video \
    --operation.type convert_image_to_video \
    --push_to_hub true

# Convert with custom video codec and quality settings
lerobot-edit-dataset \
    --repo_id lerobot/pusht_image \
    --operation.type convert_image_to_video \
    --operation.output_dir outputs/pusht_video \
    --operation.rgb_encoder.vcodec libsvtav1 \
    --operation.rgb_encoder.pix_fmt yuv420p \
    --operation.rgb_encoder.g 2 \
    --operation.rgb_encoder.crf 30

# Convert a dataset that includes depth maps, customizing the depth encoder
lerobot-edit-dataset \
    --repo_id lerobot/pusht_image \
    --operation.type convert_image_to_video \
    --operation.output_dir outputs/pusht_video \
    --operation.depth_encoder.depth_min 0.01 \
    --operation.depth_encoder.depth_max 10.0 \
    --operation.depth_encoder.use_log true

# Convert only specific episodes
lerobot-edit-dataset \
    --repo_id lerobot/pusht_image \
    --operation.type convert_image_to_video \
    --operation.output_dir outputs/pusht_video \
    --operation.episode_indices "[0, 1, 2, 5, 10]"

# Convert with multiple workers for parallel processing
lerobot-edit-dataset \
    --repo_id lerobot/pusht_image \
    --operation.type convert_image_to_video \
    --operation.output_dir outputs/pusht_video \
    --operation.num_workers 8

# For memory-constrained systems, users can now specify limits:
lerobot-edit-dataset \
    --repo_id lerobot/pusht_image \
    --operation.type convert_to_video \
    --operation.max_episodes_per_batch 50 \
    --operation.max_frames_per_batch 10000
```

**Parameters:**

- `output_dir`: Custom output directory (optional - by default uses `new_repo_id` or `{repo_id}_video`)
- `rgb_encoder`: Video encoder settings applied to RGB cameras — all sub-fields accessible via `--operation.rgb_encoder.<field>`. See [Video Encoding Parameters](./video_encoding_parameters) for more details.
- `depth_encoder`: Video encoder settings applied to depth-map cameras (e.g. from an Intel RealSense). In addition to the standard encoder fields it exposes the depth quantization knobs (`depth_min`, `depth_max`, `shift`, `use_log`), accessible via `--operation.depth_encoder.<field>`. These quantization settings are persisted to the dataset metadata so depth can be dequantized back to physical units on load. See the [Depth streams](./video_encoding_parameters#depth-streams) section for details.
- `episode_indices`: List of specific episodes to convert (default: all episodes)
- `num_workers`: Number of parallel workers for processing (default: 4)

**Note:** The resulting dataset will be a proper LeRobotDataset with all cameras encoded as videos in the `videos/` directory, with parquet files containing only metadata (no raw image data). Depth-map cameras are detected automatically and routed to the `depth_encoder`, while RGB cameras use the `rgb_encoder`. All episodes, stats, and tasks are preserved.

#### Re-encode Videos

Re-encode the videos of an existing video dataset with different encoder settings, without going back to raw frames. RGB videos use the `rgb_encoder` and depth videos use the `depth_encoder`. Provide only the encoder(s) you want to re-encode; the other stream type is left untouched.

```bash
# Re-encode all RGB videos with new settings (saves to lerobot/pusht_reencoded by default)
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --operation.type reencode_videos \
    --operation.rgb_encoder.vcodec h264 \
    --operation.rgb_encoder.pix_fmt yuv420p \
    --operation.rgb_encoder.crf 23

# Re-encode both RGB and depth videos in a dataset with depth maps
lerobot-edit-dataset \
    --repo_id lerobot/pusht_depth \
    --operation.type reencode_videos \
    --operation.rgb_encoder.vcodec h264 \
    --operation.depth_encoder.crf 50
```

**Parameters:**

- `rgb_encoder`: Encoder settings applied to every RGB video. Omit to skip re-encoding RGB videos.
- `depth_encoder`: Encoder settings applied to every depth video. Omit to skip re-encoding depth videos.
- `num_workers`: Number of parallel workers for processing.

> [!NOTE]
> When re-encoding depth videos, the existing depth quantization parameters (`depth_min`, `depth_max`, `shift`, `use_log`) and the `is_depth_map` flag are **preserved** — re-encoding only changes the codec/quality of the stored stream, not how depth is dequantized on load.

### Show the information of datasets

Show the information of datasets such as number of episode, number of frame, File size and so on.
No change will be made to the dataset

```bash

# Show dataset information without feature details
lerobot-edit-dataset \
    --repo_id lerobot/pusht_image \
    --operation.type info \

# Show dataset information with feature details
lerobot-edit-dataset \
    --repo_id lerobot/pusht_image \
    --operation.type info \
    --operation.show_features true

```

**Parameters:**

- `parameters`: The flag to control show or no show dataset information with feature details.(default=false)

### Push to Hub

Add the `--push_to_hub true` flag to any command to automatically upload the resulting dataset to the Hugging Face Hub:

```bash
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --new_repo_id lerobot/pusht_after_deletion \
    --operation.type delete_episodes \
    --operation.episode_indices "[0, 2, 5]" \
    --push_to_hub true
```

There is also a tool for adding features to a dataset that is not yet covered in `lerobot-edit-dataset`.

# Dataset Visualization

## Online Visualization

When you record a dataset using `lerobot`, it automatically uploads to the Hugging Face Hub unless you specify otherwise. To view the dataset online, use our **LeRobot Dataset Visualizer**, available at:
https://huggingface.co/spaces/lerobot/visualize_dataset

## Local Visualization

You can also visualize episodes from a dataset locally using our command-line tool.

**From the Hugging Face Hub:**

```bash
lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0
```

**From a local folder:**
Add the `--root` option and set `--mode local`. For example, to search in `./my_local_data_dir/lerobot/pusht`:

```bash
lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --root ./my_local_data_dir \
    --mode local \
    --episode-index 0
```

Once executed, the tool opens `rerun.io` and displays the camera streams, robot states, and actions for the selected episode.

To use [Foxglove](https://foxglove.dev) instead of Rerun, install the extra add `--display-mode foxglove`. This starts a WebSocket server (connect the Foxglove app to `ws://127.0.0.1:8765`) that serves the episode as a seekable timeline you can play/pause and scrub.

For advanced usage—including visualizing datasets stored on a remote server—run:

```bash
lerobot-dataset-viz --help
```
