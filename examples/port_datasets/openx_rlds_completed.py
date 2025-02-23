from pathlib import Path

import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata


def main():
    repo_id = "cadene/droid"
    datetime = "2025-02-22_11-23-54"
    port_log_dir = Path(f"/fsx/remi_cadene/logs/{datetime}_port_openx_droid")

    compl_dir = port_log_dir / "completions"

    paths = list(compl_dir.glob("*"))
    total_items = len(paths)

    # Use tqdm with the total parameter
    wrong_completions = []
    error_messages = []
    for i, path in tqdm.tqdm(enumerate(paths), total=total_items):
        try:
            rank = path.name.lstrip("0")
            if rank == "":
                rank = 0
            meta = LeRobotDatasetMetadata(f"{repo_id}_{datetime}_world_2048_rank_{rank}")
            last_episode_index = meta.total_episodes - 1
            last_ep_data_path = meta.root / meta.get_data_file_path(last_episode_index)

            if not last_ep_data_path.exists():
                raise ValueError(path)

            for vid_key in meta.video_keys:
                last_ep_vid_path = meta.root / meta.get_video_file_path(last_episode_index, vid_key)
                if not last_ep_vid_path.exists():
                    raise ValueError(path)

        except Exception as e:
            error_messages.append(str(e))
            wrong_completions.append(path)

    for path, error_msg in zip(wrong_completions, error_messages, strict=False):
        print(path)
        print(error_msg)
        print()
    #     path.unlink()

    print(f"Error {len(wrong_completions)} / {total_items}")


if __name__ == "__main__":
    main()
