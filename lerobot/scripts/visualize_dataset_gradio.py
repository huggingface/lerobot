#!/usr/bin/env python

"""Visualize data from individual LeRobot dataset episodes.

```bash
$ python lerobot/scripts/visualize_dataset_gradio.py
$ open http://127.0.0.1:7860
```

"""

import gradio as gr
import rerun as rr
import rerun.blueprint as rrb
import torch
import tqdm
from gradio_rerun import Rerun

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.video_utils import SequentialRerunVideoReader


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, episode_index):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self):
        return len(self.frame_ids)


@rr.thread_local_stream("lerobot_visualization")
def visualize_dataset(
    dataset: dict[str, LeRobotDataset],
    episode_index: int,
):
    stream = rr.binary_stream()

    batch_size = 32
    num_workers = 0

    dataset = dataset["dataset"]

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                rrb.TimeSeriesView(),
                rrb.Horizontal(contents=[rrb.Spatial2DView(origin=key) for key in dataset.camera_keys]),
            ),
            rrb.TimePanel(expanded=False),
            rrb.SelectionPanel(expanded=False),
        )
    )

    yield stream.read()

    video_reader = SequentialRerunVideoReader(dataset.repo_id, dataset.tolerance_s, compression=95)
    for key in dataset.camera_keys:
        video_reader.start_downloading(key)

    episode_sampler = EpisodeSampler(dataset, episode_index)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        # iterate over the batch
        for i in range(len(batch["index"])):
            rr.set_time_sequence("frame_index", batch["frame_index"][i].item())
            rr.set_time_seconds("timestamp", batch["timestamp"][i].item())

            # display each camera image
            for key in dataset.camera_keys:
                img = video_reader.next_frame(batch[key]["path"][i], batch[key]["timestamp"][i])
                if img is not None:
                    rr.log(key, img)

            # display each dimension of action space (e.g. actuators command)
            if "action" in batch:
                for dim_idx, val in enumerate(batch["action"][i]):
                    rr.log(f"action/{dim_idx}", rr.Scalar(val.item()))

            # display each dimension of observed state space (e.g. agent position in joint space)
            if "observation.state" in batch:
                for dim_idx, val in enumerate(batch["observation.state"][i]):
                    rr.log(f"state/{dim_idx}", rr.Scalar(val.item()))

            if "next.done" in batch:
                rr.log("next.done", rr.Scalar(batch["next.done"][i].item()))

            if "next.reward" in batch:
                rr.log("next.reward", rr.Scalar(batch["next.reward"][i].item()))

            if "next.success" in batch:
                rr.log("next.success", rr.Scalar(batch["next.success"][i].item()))

            yield stream.read()


def update_episodes(dataset, loaded_dataset):
    loaded_dataset["dataset"] = LeRobotDataset(dataset)
    dataset = loaded_dataset["dataset"]
    return gr.update(choices=list(range(dataset.num_episodes)), value=0)


def main():
    with gr.Blocks() as demo:
        loaded_dataset = gr.State({})
        with gr.Row():
            with gr.Column():
                dataset = gr.Dropdown(choices=lerobot.available_real_world_datasets)
            with gr.Column():
                episode = gr.Dropdown(choices=[], interactive=True)
        with gr.Row():
            viewer = Rerun(streaming=True, height=800)

        dataset.change(update_episodes, inputs=[dataset, loaded_dataset], outputs=[episode])
        episode.change(visualize_dataset, inputs=[loaded_dataset, episode], outputs=[viewer])

    demo.queue(default_concurrency_limit=10)
    demo.launch()


if __name__ == "__main__":
    main()
