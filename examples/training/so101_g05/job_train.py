"""Use Trackio with the older G05 training entry point's small logger interface."""

import atexit
import os

import trackio


class TrackioBridge:
    def __init__(self, cfg):
        self.run = trackio.init(
            project="so101-g05",
            name=os.environ["G05_RUN_NAME"],
            space_id="maximellerbach/so101-fineart-tracking",
            private=True,
            config=cfg.to_dict(),
            resume="never",
            capture_console=True,
        )
        print(f"G05 Trackio run: {os.environ['G05_RUN_NAME']}", flush=True)
        atexit.register(trackio.finish)

    def log_dict(self, values, step=None, mode="train", custom_step_key=None):
        data = {
            f"{mode}/{key}": value for key, value in values.items() if isinstance(value, (int, float, str))
        }
        data["train/optimizer_step"] = step // int(os.environ.get("G05_ACCUMULATION", "1"))
        trackio.log(data, step=step)

    def log_policy(self, checkpoint_dir):
        # Model checkpoints go directly to the public model repo, not the private dashboard.
        pass

    def log_video(self, video_path, step, mode="eval"):
        trackio.log({f"{mode}/video": trackio.Video(video_path)}, step=step)


if __name__ == "__main__":
    import lerobot.scripts.lerobot_train as training

    training.WandBLogger = TrackioBridge
    training.main()
