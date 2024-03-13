# TODO(rcadene): add tests
# TODO(rcadene): what is the best format to store/load videos?

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Sequence

import einops
import torch
import torchaudio
import torchrl
from matplotlib import pyplot as plt
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import dispatch
from tensordict.utils import NestedKey
from torchaudio.io import StreamReader
from torchrl.data.replay_buffers.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler, SliceSamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import TensorStorage, _collate_id
from torchrl.data.replay_buffers.writers import ImmutableDatasetWriter, Writer
from torchrl.envs.transforms import Transform
from torchrl.envs.transforms.transforms import Compose

from lerobot.common.utils import set_seed

NUM_STATE_CHANNELS = 12
NUM_ACTION_CHANNELS = 12


def yuv_to_rgb(frames):
    assert frames.dtype == torch.uint8
    assert frames.ndim == 4
    assert frames.shape[1] == 3

    frames = frames.cpu().to(torch.float)
    y = frames[..., 0, :, :]
    u = frames[..., 1, :, :]
    v = frames[..., 2, :, :]

    y /= 255
    u = u / 255 - 0.5
    v = v / 255 - 0.5

    r = y + 1.14 * v
    g = y + -0.396 * u - 0.581 * v
    b = y + 2.029 * u

    rgb = torch.stack([r, g, b], 1)
    rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
    return rgb


def count_frames(video_path):
    try:
        # Construct the ffprobe command to get the number of frames
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            video_path,
        ]

        # Execute the ffprobe command and capture the output
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Convert the output to an integer
        num_frames = int(result.stdout.strip())

        return num_frames
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1


def get_frame_rate(video_path):
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            video_path,
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # The frame rate is typically represented as a fraction (e.g., "30000/1001").
        # To convert it to a float, we can evaluate the fraction.
        frame_rate = eval(result.stdout.strip())

        return frame_rate
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1


def get_frame_timestamps(frame_rate, num_frames):
    timestamps = [(1 / frame_rate) * i for i in range(num_frames)]
    return timestamps


# class ClearDeviceTransform(Transform):
#     invertible = False

#     def __init__(self):
#         super().__init__()

#     def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
#         # _reset is called once when the environment reset to normalize the first observation
#         tensordict_reset = self._call(tensordict_reset)
#         return tensordict_reset

#     @dispatch(source="in_keys", dest="out_keys")
#     def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
#         return self._call(tensordict)

#     def _call(self, td: TensorDictBase) -> TensorDictBase:
#         td.clear_device_()
#         return td


class ViewSliceHorizonTransform(Transform):
    invertible = False

    def __init__(self, num_slices, horizon):
        super().__init__()
        self.num_slices = num_slices
        self.horizon = horizon

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        # _reset is called once when the environment reset to normalize the first observation
        tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    @dispatch(source="in_keys", dest="out_keys")
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict)

    def _call(self, td: TensorDictBase) -> TensorDictBase:
        td = td.view(self.num_slices, self.horizon)
        return td


class KeepFrames(Transform):
    invertible = False

    def __init__(
        self,
        positions,
        in_keys: Sequence[NestedKey],
        out_keys: Sequence[NestedKey] = None,
    ):
        if isinstance(positions, list):
            assert isinstance(positions[0], int)
        # TODO(rcadene)L add support for `isinstance(positions, int)`?

        self.positions = positions
        if out_keys is None:
            out_keys = in_keys
        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        # _reset is called once when the environment reset to normalize the first observation
        tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    @dispatch(source="in_keys", dest="out_keys")
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict)

    def _call(self, td: TensorDictBase) -> TensorDictBase:
        # we need set batch_size=[] before assigning a different shape to td[outkey]
        td.batch_size = []

        for inkey, outkey in zip(self.in_keys, self.out_keys, strict=False):
            # TODO(rcadene): don't know how to do `inkey not in td`
            if td.get(inkey, None) is None:
                continue
            td[outkey] = td[inkey][:, self.positions]
        return td


class DecodeVideoTransform(Transform):
    invertible = False

    def __init__(
        self,
        device="cpu",
        # format options are None=yuv420p (usually), rgb24, bgr24, etc.
        format: str | None = None,
        frame_rate: int | None = None,
        width: int | None = None,
        height: int | None = None,
        in_keys: Sequence[NestedKey] = None,
        out_keys: Sequence[NestedKey] = None,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
    ):
        self.device = device
        self.format = format
        self.frame_rate = frame_rate
        self.width = width
        self.height = height
        self.video_id_to_path = None
        if out_keys is None:
            out_keys = in_keys
        if in_keys_inv is None:
            in_keys_inv = out_keys
        if out_keys_inv is None:
            out_keys_inv = in_keys
        super().__init__(
            in_keys=in_keys, out_keys=out_keys, in_keys_inv=in_keys_inv, out_keys_inv=out_keys_inv
        )

    def set_video_id_to_path(self, video_id_to_path):
        self.video_id_to_path = video_id_to_path

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        # _reset is called once when the environment reset to normalize the first observation
        tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    @dispatch(source="in_keys", dest="out_keys")
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict)

    def _call(self, td: TensorDictBase) -> TensorDictBase:
        assert (
            self.video_id_to_path is not None
        ), "Setting a video_id_to_path dictionary with `self.set_video_id_to_path(video_id_to_path)` is required."

        for inkey, outkey in zip(self.in_keys, self.out_keys, strict=False):
            # TODO(rcadene): don't know how to do `inkey not in td`
            if td.get(inkey, None) is None:
                continue

            bsize = len(td[inkey])  # num episodes in the batch
            b_frames = []
            for i in range(bsize):
                assert (
                    td["observation", "frame", "video_id"].ndim == 2
                ), "We expect 2 dims. Respectively, number of episodes in the batch and number of observations"

                ep_video_ids = td[inkey]["video_id"][i]
                timestamps = td[inkey]["timestamp"][i]
                frame_ids = td["frame_id"][i]

                unique_video_id = (ep_video_ids.min() == ep_video_ids.max()).item()
                assert unique_video_id

                is_ascending = torch.all(timestamps[:-1] <= timestamps[1:]).item()
                assert is_ascending

                is_contiguous = ((frame_ids[1:] - frame_ids[:-1]) == 1).all().item()
                assert is_contiguous

                FIRST_FRAME = 0  # noqa: N806
                video_id = ep_video_ids[FIRST_FRAME].item()
                video_path = self.video_id_to_path[video_id]
                first_frame_ts = timestamps[FIRST_FRAME].item()
                num_contiguous_frames = len(timestamps)

                filter_desc = []
                video_stream_kwgs = {
                    "frames_per_chunk": num_contiguous_frames,
                    "buffer_chunk_size": num_contiguous_frames,
                }

                # choice of decoder
                if self.device == "cuda":
                    video_stream_kwgs["hw_accel"] = "cuda"
                    video_stream_kwgs["decoder"] = "h264_cuvid"
                else:
                    video_stream_kwgs["decoder"] = "h264"

                # resize
                resize_width = self.width is not None
                resize_height = self.height is not None
                if resize_width or resize_height:
                    if self.device == "cuda":
                        assert resize_width and resize_height
                        video_stream_kwgs["decoder_option"] = {"resize": f"{self.width}x{self.height}"}
                    else:
                        scales = []
                        if resize_width:
                            scales.append(f"width={self.width}")
                        if resize_height:
                            scales.append(f"height={self.height}")
                        filter_desc.append(f"scale={':'.join(scales)}")

                # choice of format
                if self.format is not None:
                    if self.device == "cuda":
                        # TODO(rcadene): rebuild ffmpeg with --enable-cuda-nvcc, --enable-cuvid, and --enable-libnpp
                        raise NotImplementedError()
                        # filter_desc = f"scale=format={self.format}"
                        # filter_desc = f"scale_cuda=format={self.format}"
                        # filter_desc = f"scale_npp=format={self.format}"
                    else:
                        filter_desc.append(f"format=pix_fmts={self.format}")

                # choice of frame rate
                if self.frame_rate is not None:
                    filter_desc.append(f"fps={self.frame_rate}")

                if len(filter_desc) > 0:
                    video_stream_kwgs["filter_desc"] = ",".join(filter_desc)

                # create a stream and load a certain number of frame at a certain frame rate
                # TODO(rcadene): make sure it's the most optimal way to do it
                s = StreamReader(video_path)
                s.seek(first_frame_ts)
                s.add_video_stream(**video_stream_kwgs)
                s.fill_buffer()
                (frames,) = s.pop_chunks()

                b_frames.append(frames)

            td[outkey] = torch.stack(b_frames)

        if self.device == "cuda":
            # make sure we return a cuda tensor, since the frames can be unwillingly sent to cpu
            assert "cuda" in str(td[outkey].device), f"{td[outkey].device} instead of cuda"
        return td


class VideoExperienceReplay(TensorDictReplayBuffer):
    def __init__(
        self,
        batch_size: int = None,
        *,
        root: Path = None,
        pin_memory: bool = False,
        prefetch: int = None,
        sampler: SliceSampler = None,
        collate_fn: Callable = None,
        writer: Writer = None,
        transform: "torchrl.envs.Transform" = None,
    ):
        self.data_dir = root / "2024_03_17_test_dataset"
        self.rb_dir = self.data_dir / "replay_buffer"

        storage, meta_data = self._load_or_download()

        # hack to access video paths
        assert isinstance(transform, Compose)
        for tf in transform:
            if isinstance(tf, DecodeVideoTransform):
                tf.set_video_id_to_path(meta_data["video_id_to_path"])

        super().__init__(
            storage=storage,
            sampler=sampler,
            writer=ImmutableDatasetWriter() if writer is None else writer,
            collate_fn=_collate_id if collate_fn is None else collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            batch_size=batch_size,
            transform=transform,
        )

    def _load_or_download(self, force_download=False):
        if not force_download and self.data_dir.exists():
            storage = TensorStorage(TensorDict.load_memmap(self.rb_dir))
            meta_data = torch.load(self.data_dir / "meta_data.pth")
        else:
            storage, meta_data = self._download()
            torch.save(meta_data, self.data_dir / "meta_data.pth")

        # required to not send cuda frames to cpu by default
        storage._storage.clear_device_()
        return storage, meta_data

    def _download(self):
        num_episodes = 1
        video_id_to_path = {}
        for episode_id in range(num_episodes):
            video_path = torchaudio.utils.download_asset(
                "tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"
            )
            # several episodes can belong to the same video
            video_id = episode_id
            video_id_to_path[video_id] = video_path

            print(f"{video_path=}")
            num_frames = count_frames(video_path)
            print(f"{num_frames=}")
            frame_rate = get_frame_rate(video_path)
            print(f"{frame_rate=}")

            frame_timestamps = get_frame_timestamps(frame_rate, num_frames)

            reward = torch.zeros(num_frames, 1, dtype=torch.float32)
            success = torch.zeros(num_frames, 1, dtype=torch.bool)
            done = torch.zeros(num_frames, 1, dtype=torch.bool)
            state = torch.randn(num_frames, NUM_STATE_CHANNELS, dtype=torch.float32)
            action = torch.randn(num_frames, NUM_ACTION_CHANNELS, dtype=torch.float32)
            timestamp = torch.tensor(frame_timestamps)
            frame_id = torch.arange(0, num_frames, 1)
            episode_id_tensor = torch.tensor([episode_id] * num_frames, dtype=torch.int)
            video_id_tensor = torch.tensor([video_id] * num_frames, dtype=torch.int)

            # last step of demonstration is considered done
            done[-1] = True

            ep_td = TensorDict(
                {
                    ("observation", "frame", "video_id"): video_id_tensor[:-1],
                    ("observation", "frame", "timestamp"): timestamp[:-1],
                    ("observation", "state"): state[:-1],
                    "action": action[:-1],
                    "episode": episode_id_tensor[:-1],
                    "frame_id": frame_id[:-1],
                    ("next", "observation", "frame", "video_id"): video_id_tensor[1:],
                    ("next", "observation", "frame", "timestamp"): timestamp[1:],
                    ("next", "observation", "state"): state[1:],
                    ("next", "reward"): reward[1:],
                    ("next", "done"): done[1:],
                    ("next", "success"): success[1:],
                },
                batch_size=num_frames - 1,
            )

            # TODO:
            total_frames = num_frames - 1

            if episode_id == 0:
                # hack to initialize tensordict data structure to store episodes
                td_data = ep_td[0].expand(total_frames).memmap_like(self.rb_dir)

            td_data[:] = ep_td

        meta_data = {
            "video_id_to_path": video_id_to_path,
        }

        return TensorStorage(td_data.lock_()), meta_data


if __name__ == "__main__":
    import time

    import tqdm

    def create_replay_buffer(device):
        num_slices = 1
        horizon = 2
        batch_size = num_slices * horizon

        sampler = SliceSamplerWithoutReplacement(
            num_slices=num_slices,
            strict_length=True,
            shuffle=False,
        )

        transforms = [
            # ClearDeviceTransform(),
            ViewSliceHorizonTransform(num_slices, horizon),
            KeepFrames(positions=[0], in_keys=[("observation")]),
            DecodeVideoTransform(
                device=device,
                frame_rate=None,
                in_keys=[("observation", "frame")],
                out_keys=[("observation", "frame", "data")],
            ),
        ]

        replay_buffer = VideoExperienceReplay(
            root=Path("tmp"),
            batch_size=batch_size,
            # prefetch=4,
            transform=Compose(*transforms),
            sampler=sampler,
        )
        return replay_buffer

    def test_time():
        replay_buffer = create_replay_buffer(device="cuda")

        start = time.time()
        for _ in tqdm.tqdm(range(2)):
            # include_info=False is required to not have a batch_size mismatch error with the truncated key (2,8) != (16, 1)
            replay_buffer.sample(include_info=False)
        torch.cuda.synchronize()
        print(time.time() - start)

        start = time.time()
        for _ in tqdm.tqdm(range(10)):
            replay_buffer.sample(include_info=False)
        torch.cuda.synchronize()
        print(time.time() - start)

    def test_plot():
        rb_cuda = create_replay_buffer(device="cuda")
        rb_cpu = create_replay_buffer(device="cpu")

        n_rows = 2  # len(replay_buffer)
        fig, axes = plt.subplots(n_rows, 2, figsize=[12.8, 16.0])
        for i in range(n_rows):
            set_seed(1337 + i)
            batch_cpu = rb_cpu.sample(include_info=False)
            print(batch_cpu["frame_id"])
            frames = batch_cpu["observation", "frame", "data"]

            frames = einops.rearrange(frames, "b t c h w -> (b t) c h w")
            frames = yuv_to_rgb(frames)
            frames = einops.rearrange(frames, "bt c h w -> bt h w c")

            assert frames.shape[0] == 1
            axes[i][0].imshow(frames[0])

            set_seed(1337 + i)
            batch_cuda = rb_cuda.sample(include_info=False)
            print(batch_cuda["frame_id"])
            frames = batch_cuda["observation", "frame", "data"]

            frames = einops.rearrange(frames, "b t c h w -> (b t) c h w")
            frames = yuv_to_rgb(frames)
            frames = einops.rearrange(frames, "bt c h w -> bt h w c")

            assert frames.shape[0] == 1
            axes[i][1].imshow(frames[0])

        axes[0][0].set_title("Software decoder")
        axes[0][1].set_title("HW decoder")
        plt.setp(axes, xticks=[], yticks=[])
        plt.tight_layout()
        fig.savefig(rb_cuda.data_dir / "test.png", dpi=300)

    # test_time()
    test_plot()
