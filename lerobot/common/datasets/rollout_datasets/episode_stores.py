from pathlib import Path

import numcodecs
import numpy as np
import torch
import zarr

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.rollout_datasets.zarr_utils import to_hf_dataset, read_data_from_zarr
from lerobot.common.datasets.utils import calculate_episode_data_index

import h5py

def _append_episode_to_hdf5(ep_dict, hdf5_file_path):
    with h5py.File(hdf5_file_path, 'a') as hdf5_file:
        for key, value in ep_dict.items():
            # Convert lists of dictionaries or strings to JSON strings
            if isinstance(value[0], dict) or isinstance(value[0], str):
                value = np.array([str(v) for v in value], dtype=h5py.string_dtype(encoding='utf-8'))

            if key not in hdf5_file:
                data_shape = (0, *np.array(value).shape[1:])
                maxshape = (None, *data_shape[1:])
                
                # Create a dataset with gzip compression
                hdf5_file.create_dataset(
                    key,
                    shape=data_shape,
                    maxshape=maxshape,
                    dtype=value.dtype,
                    chunks=True,
                    compression='gzip',
                    compression_opts=4  # Adjust compression level as needed
                )

            # Resize the dataset to accommodate new data
            current_shape = hdf5_file[key].shape
            new_shape = (current_shape[0] + len(value), *current_shape[1:])
            hdf5_file[key].resize(new_shape)
            
            # Append new data
            hdf5_file[key][-len(value):] = value


def _append_episode_to_zarr(ep_dict, zarr_group):
    for key, value in ep_dict.items():
        if key not in zarr_group:
            data_shape = (0, *np.array(value).shape[1:])

            object_codec = numcodecs.JSON() if isinstance(value[0], dict) or isinstance(value[0], str) else None
            compressor = zarr.Blosc(cname='zstd', clevel=5) 

            zarr_group.create_dataset(
                key,
                shape=data_shape,
                chunks=(1, *data_shape[1:]),
                dtype=np.array(value).dtype,
                maxshape=(None, *data_shape[1:]),
                object_codec=object_codec,
                compressor=compressor
            )

        zarr_group[key].append(value)


class EpisodeVideoStore(object):
    def __init__(self, root_group: zarr.Group, info: dict = None):
        super().__init__()
        self.root_group = root_group
        if 'info' in self.root_group.attrs and 'data' in self.root_group:
            print("Connected to the exising zarr")
            self.data_group = self.root_group['data']
        else:
            print("Create a new zarr")
            self.data_group = self.root_group.create_group('data')

        if info is not None:
            self.root_group.attrs['info'] = info
        if 'num_episodes' not in self.root_group.attrs:
            self.root_group.attrs['num_episodes'] = 0

    def add_episode(self, episode_dict):
        assert 'episode_index' not in episode_dict
        assert 'frame_index' in episode_dict

        episode_dict['episode_index'] = torch.tensor([self.num_episodes] * len(episode_dict['frame_index']),
                                                     dtype=torch.int64)
        episode_dict = self._cleansing_episode_dict(episode_dict)
        _append_episode_to_zarr(episode_dict, self.data_group)
        self.root_group.attrs['num_episodes'] += 1
        print(f"Successfully added episode: {self.root_group.attrs['num_episodes']}")

    def convert_to_lerobot_dataset(self, repo_id, **kwargs):
        zarr_dict = read_data_from_zarr(self.data_group)
        hf_dataset = to_hf_dataset(zarr_dict)
        episode_data_index = calculate_episode_data_index(hf_dataset)

        info = self.info
        return LeRobotDataset.from_preloaded(
            repo_id=repo_id,
            hf_dataset=hf_dataset,
            episode_data_index=episode_data_index,
            info=info,
            videos_dir=Path(info['videos_dir']),
            **kwargs
        )

    @classmethod
    def create_from_path(cls, root_path, info: dict = None, mode='r'):
        assert mode == 'r' or 'videos_dir' in info
        root_group = zarr.open(root_path, mode=mode)
        return EpisodeVideoStore(root_group, info)

    @property
    def info(self):
        return self.root_group.attrs.get('info')

    @property
    def num_episodes(self):
        return self.root_group.attrs['num_episodes']

    def _cleansing_episode_dict(self, episode_dict):
        return {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in episode_dict.items()}

class EpisodeVideoStoreAsHDF5(object):
    def __init__(self, hdf5_file_path: str, info: dict = None):
        self.hdf5_file_path = hdf5_file_path
        with h5py.File(self.hdf5_file_path, 'a') as hdf5_file:
            if 'info' in hdf5_file.attrs and 'data' in hdf5_file:
                print("Connected to the existing HDF5 file")
            else:
                print("Creating a new HDF5 file")
                hdf5_file.create_group('data')

            if info is not None:
                import json
                hdf5_file.attrs['info'] = json.dumps(info)
            if 'num_episodes' not in hdf5_file.attrs:
                hdf5_file.attrs['num_episodes'] = 0

    def add_episode(self, episode_dict):
        assert 'episode_index' not in episode_dict
        assert 'frame_index' in episode_dict

        with h5py.File(self.hdf5_file_path, 'a') as hdf5_file:
            episode_dict['episode_index'] = torch.tensor(
                [self.num_episodes] * len(episode_dict['frame_index']),
                dtype=torch.int64
            )
            episode_dict = self._cleansing_episode_dict(episode_dict)
            _append_episode_to_hdf5(episode_dict, self.hdf5_file_path)
            hdf5_file.attrs['num_episodes'] += 1
            print(f"Successfully added episode: {hdf5_file.attrs['num_episodes']}")

    def convert_to_lerobot_dataset(self, repo_id, **kwargs):
        with h5py.File(self.hdf5_file_path, 'r') as hdf5_file:
            zarr_dict = {key: hdf5_file['data'][key][:] for key in hdf5_file['data']}
            hf_dataset = to_hf_dataset(zarr_dict)
            episode_data_index = calculate_episode_data_index(hf_dataset)

            info = self.info
            return LeRobotDataset.from_preloaded(
                repo_id=repo_id,
                hf_dataset=hf_dataset,
                episode_data_index=episode_data_index,
                info=info,
                videos_dir=Path(info['videos_dir']),
                **kwargs
            )

    @classmethod
    def create_from_path(cls, hdf5_file_path, info: dict = None):
        return EpisodeVideoStoreAsHDF5(hdf5_file_path, info)

    @property
    def info(self):
        with h5py.File(self.hdf5_file_path, 'r') as hdf5_file:
            return hdf5_file.attrs.get('info')

    @property
    def num_episodes(self):
        with h5py.File(self.hdf5_file_path, 'r') as hdf5_file:
            return hdf5_file.attrs['num_episodes']

    def _cleansing_episode_dict(self, episode_dict):
        return {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in episode_dict.items()}


    def read_hdf5_file(hdf5_file_path):
        with h5py.File(hdf5_file_path, 'r') as hdf5_file:
            # 데이터셋 이름을 출력
            print("Datasets in the file:")
            for name in hdf5_file:
                print(name)
            
            # 특정 데이터셋 읽기
            if 'action' in hdf5_file:
                data_group = hdf5_file['data']
                for dataset_name in data_group:
                    dataset = data_group[dataset_name]
                    print(f"Dataset {dataset_name}:")
                    print(dataset[:])  # 데이터셋의 모든 데이터를 출력

