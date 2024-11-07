import torch
import zarr
import numpy as np
import numcodecs
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from lerobot.common.datasets.utils import calculate_episode_data_index

from lerobot.common.datasets.push_dataset_to_hub.pusht_zarr_format import to_hf_dataset


def _append_episode_to_zarr(ep_dict, zarr_group):
    for key, value in ep_dict.items():
        if key not in zarr_group:
            data_shape = (0, *np.array(value).shape[1:])

            object_codec = numcodecs.JSON() if isinstance(value[0], dict) or isinstance(value[0], str) else None
            zarr_group.create_dataset(
                key,
                shape=data_shape,
                chunks=(1, *data_shape[1:]),
                dtype=np.array(value).dtype,
                maxshape=(None, *data_shape[1:]),
                object_codec=object_codec
            )

        zarr_group[key].append(value)


def _read_dataset_from_zarr(zarr_group):
    data_dict = {}
    for k, v in zarr_group.items():
        data_dict[k] = v if not isinstance(v, np.ndarray) else torch.from_numpy(v)
    return data_dict


class EpisodeVideoStore(object):
    def __init__(self, root_group: zarr.Group, info: dict = None, num_episodes: int = None):
        super().__init__()
        self.root_group = root_group
        self.root_group.attrs.put('info', info)
        self._num_episodes = num_episodes if num_episodes else self._get_num_episodes()

    def add_episode(self, episode_dict):
        assert self._num_episodes <= episode_dict['episode_index'][0]

        episode_dict = self._cleansing_episode_dict(episode_dict)
        _append_episode_to_zarr(episode_dict, self.root_group)
        self._num_episodes += 1

    def convert_to_lerobot_dataset(self):
        # TODO: large data
        data_dict = _read_dataset_from_zarr(self.root_group)
        hf_dataset = to_hf_dataset(data_dict, video=True, keypoints_instead_of_image=False)
        episode_data_index = calculate_episode_data_index(hf_dataset)

        info = self.root_group.attrs.asdict()
        return LeRobotDataset.from_preloaded(
            repo_id='place/holder',
            hf_dataset=hf_dataset,
            episode_data_index=episode_data_index,
            info=info,
            videos_dir=info['videos_dir'],
        )

    @classmethod
    def create_from_path(cls, root_path, info: dict = None, mode='r'):
        assert 'video_dir' in info
        root_group = zarr.open(root_path, mode=mode)
        return EpisodeVideoStore(root_group, info)

    @property
    def info(self):
        return self.root_group.attrs.get('info')

    def _cleansing_episode_dict(self, episode_dict):
        return {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in episode_dict.items()}

    def _get_num_episodes(self):
        episode_index_dataset = self.root_group['episode_index']
        total_size = episode_index_dataset.shape[0]
        chunk_size = episode_index_dataset.chunks[0]

        max_episode_index = -1
        for idx in range(0, total_size, chunk_size):
            end_idx = max(idx + chunk_size, total_size - 1)
            episode_index_chunk = episode_index_dataset[idx:end_idx]
            chunk_max_episode_index = np.max(episode_index_chunk)
            if chunk_max_episode_index > max_episode_index:
                max_episode_index = chunk_max_episode_index
        return max_episode_index + 1
