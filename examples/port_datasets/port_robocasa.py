"""
Script for converting RoboCasa HDF5 datasets to LeRobot format.

This script automatically discovers image names, state/action shapes, and other properties
from the RoboCasa HDF5 dataset files. It ensures ALL keys from the original dataset are
discovered and included by:
- Checking multiple episodes (up to 10) during discovery to find all possible keys
- Loading all episode-level datasets and groups (not just obs and actions)
- Processing all observation keys found in each episode (not just those from the first episode)
- Preserving all episode attributes and metadata

Usage:
    The dataset path should be the path to the dataset with images in it (see robocasa for details).
    python examples/port_datasets/port_robocasa.py --dataset_path /path/to/dataset.hdf5 --repo_name your_hf_username/robocasa_dataset

If you want to push your dataset to the Hugging Face Hub:
    python examples/port_datasets/port_robocasa.py --dataset_path /path/to/dataset.hdf5 --repo_name your_hf_username/robocasa_dataset --push_to_hub

You can also process multiple datasets:
    python examples/port_datasets/port_robocasa.py --dataset_path /path/to/dataset1.hdf5 /path/to/dataset2.hdf5 --repo_name your_hf_username/robocasa_combined

Note: The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
"""

import os
import copy
import shutil
import h5py
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from termcolor import colored
from typing import List, Dict, Any, Union, Tuple

from lerobot.utils.constants import ACTION, OBS_STATE, OBS_IMAGES
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import encode_video_frames
import tyro

from huggingface_hub import login
HF_TOKEN = os.environ["HF_TOKEN"]
login(HF_TOKEN)

import torch
from transformers import CLIPTokenizer, CLIPTextModel

def remove_unsupported_dtypes(state_keys: List[str], state_shapes: Dict[str, tuple], state_dtypes: Dict[str, str]) -> Tuple[List[str], Dict[str, tuple], Dict[str, str]]:
    new_state_keys = []
    new_state_shapes = {}
    new_state_dtypes = {}
    for key, shape in state_shapes.items():
        if state_dtypes[key] != "object":
            new_state_keys.append(key)
            new_state_shapes[key] = shape
            new_state_dtypes[key] = state_dtypes[key]
    return new_state_keys, new_state_shapes, new_state_dtypes

def get_clip_embedding(text, tokenizer, model, device: torch.device, max_length=77):
    """
    Generate CLIP text embedding for a given text using Hugging Face transformers.
    
    Args:
        text (str): Input text to encode
        tokenizer: CLIP tokenizer
        model: CLIP text model
        device: Device to run the model on
        max_length (int): Maximum sequence length for CLIP tokenization
        
    Returns:
        np.ndarray: CLIP text embedding vector
    """
    # Tokenize the text using CLIP's tokenizer
    tokens = tokenizer(
        text=text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True
    ).to(device)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**tokens)
        # Use the pooled output (equivalent to CLIP's text features)
        text_features = outputs.pooler_output
        # Normalize the features (CLIP standard practice)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        embedding = text_features.detach().cpu().numpy()
    
    return embedding.squeeze()

def get_max_dims(hdf5_paths: List[str], keys_with_variable_length: List[str]) -> Dict[str, int]:
    max_dims = {}
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as f:
            demo_keys = list(f['data'].keys())
            for demo_key in demo_keys:
                demo_group = f['data'][demo_key]
                for key in keys_with_variable_length:
                    if key not in max_dims:
                        max_dims[key] = (0,)
                    if key not in demo_group:
                        continue
                    data = demo_group[key][:][0] # take the first timestep
                    max_dims[key] = (max(max_dims[key][0], data.shape[0]),) + tuple(data.shape[1:])
    return max_dims


def discover_dataset_properties(hdf5_paths: List[str]) -> Dict[str, Any]:
    """
    Discover image names, state/action shapes, and other properties from a RoboCasa HDF5 dataset.
    Checks ALL episodes to ensure all keys are discovered.
    
    Returns:
        Dictionary containing:
            - image_names: List of image observation names (without _image suffix)
            - image_shapes: Dict mapping image names to their shapes
            - state_keys: List of non-image observation keys that are state features
            - state_shapes: Dict mapping state keys to their shapes
            - action_shape: Shape of action array
            - other_keys: Dict mapping other episode-level dataset keys to their shapes/dtypes
    """
    # TODO: estimate the variable lengths keys directly from the dataset instead of hardcoding them
    keys_with_variable_length = ['states', 'obs/object-state', 'obs/objects-joint-state', 'obs/object']
    max_dims = get_max_dims(hdf5_paths, keys_with_variable_length)
    hdf5_path = hdf5_paths[0]
    with h5py.File(hdf5_path, "r") as f:
        # Get all episodes to discover structure (check multiple episodes to find all keys)
        demos = sorted(list(f["data"].keys()))
        if not demos:
            raise ValueError(f"No episodes found in {hdf5_path}")
        
        # Discover image observation keys and state keys from ALL episodes
        image_names = set()
        image_shapes = {}
        state_keys = set()
        state_shapes = {}
        state_dtypes = {}
        
        # Track other episode-level datasets and groups
        other_keys = {}  # Dict of key -> (shape, dtype)
        
        # Check multiple episodes to ensure we find all possible keys
        episodes_to_check = demos[:min(10, len(demos))]  # Check up to 10 episodes
        
        for ep_name in episodes_to_check:
            ep_group = f["data"][ep_name]
            
            # Discover all keys in the episode group
            for key in ep_group.keys():
                item = ep_group[key]
                if isinstance(item, h5py.Group):
                    # It's a group (like "obs", "action_infos", etc.)
                    if key == "obs":
                        # Process observation group
                        obs_group = item
                        for obs_key in obs_group.keys():
                            if obs_key.endswith("_image"):
                                # Remove "_image" suffix for the feature name
                                image_name = obs_key[:-6]  # Remove "_image"
                                image_names.add(image_name)
                                if image_name not in image_shapes:
                                    image_shapes[image_name] = obs_group[obs_key].shape[1:]  # Skip time dimension
                            elif f'obs/{obs_key}' in max_dims:
                                state_keys.add(obs_key)
                                if obs_key not in state_shapes:
                                    state_shapes[obs_key] = max_dims[f'obs/{obs_key}']  # Skip time dimension
                                    state_dtypes[obs_key] = obs_group[obs_key].dtype
                            else:
                                # This might be state/proprioceptive data
                                state_keys.add(obs_key)
                                if obs_key not in state_shapes:
                                    state_shapes[obs_key] = obs_group[obs_key].shape[1:]  # Skip time dimension
                                    state_dtypes[obs_key] = obs_group[obs_key].dtype
                    else:
                        print(colored(f"Warning: Other group {key} found in episode {ep_name}, but not handled.", "yellow"))
                        # Other groups - we'll handle them separately
                        pass
                elif isinstance(item, h5py.Dataset):
                    # It's a dataset (like "actions", "states", "policy_mode", etc.)
                    if (key not in ["actions"]) and (key not in other_keys):  # Skip actions as it's handled separately
                        if key in max_dims:
                            other_keys[key] = (max_dims[key], str(item.dtype))
                        else:
                            other_keys[key] = (item.shape[1:], str(item.dtype)) if len(item.shape) > 1 else ((1,), str(item.dtype))
        
        # remove keys whose dtype are dobjects
        state_keys, state_shapes, state_dtypes = remove_unsupported_dtypes(state_keys, state_shapes, state_dtypes)

        # Get action shape from first episode
        first_ep = demos[0]
        action_shape = f["data"][first_ep]["actions"].shape[1:]
        action_dtype = f["data"][first_ep]["actions"].dtype
        
        # Try to get FPS from metadata
        fps = None
        if "data" in f and "env_args" in f["data"].attrs:
            env_args = f["data"].attrs["env_args"]
            if env_args:
                if isinstance(env_args, bytes):
                    env_args = env_args.decode("utf-8")
                if isinstance(env_args, str):
                    env_args = json.loads(env_args)
                if 'control_freq' not in env_args:
                    print(colored(f"Warning: control_freq not found in env_args for {hdf5_path}. Defaulting to 20 fps.", "red"))
                fps = env_args.get("control_freq", 20) # default to 20 fps as in robosuite if not found
                fps = int(fps)
        
        return {
            "image_names": sorted(list(image_names)),
            "image_shapes": image_shapes,
            "state_keys": sorted(list(state_keys)),  # Sort for consistency
            "state_shapes": state_shapes,
            "state_dtypes": state_dtypes,
            "action_shape": action_shape,
            "action_dtype": action_dtype,
            "other_keys": other_keys,  # Other episode-level datasets
            "fps": fps,
            "max_dims": max_dims,
        }


def create_lerobot_features(properties: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create features dictionary for LeRobotDataset from discovered properties.
    
    Args:
        properties: Dictionary from discover_dataset_properties
    
    Returns:
        Features dictionary for LeRobotDataset.create
    """
    features = {}
    
    # Add all images as features with their original names
    image_names = properties["image_names"]
    image_shapes = properties["image_shapes"]
    
    if not image_names:
        raise ValueError("No image observations found in dataset!")
    
    # Add all images with their original names
    for img_name in image_names:
        channel_first = False
        names = ["height", "width", "channel"]
        if image_shapes[img_name][0] == 3:
            channel_first = True
            names = ["channel", "height", "width"]
        features[OBS_IMAGES + "." + img_name] = {
            "dtype": "video",
            "shape": image_shapes[img_name],
            "names": names,
        }
    
    # Add each state key as a separate feature
    state_shapes = properties["state_shapes"]
    state_dtypes = properties["state_dtypes"]
    for state_key in properties["state_keys"]:
        features[state_key] = {
            "dtype": str(state_dtypes[state_key]),
            "shape": state_shapes[state_key],
            "names": [state_key],
        }

    # Combine robot0_joint_pos_cos, robot0_joint_pos_sin and robot0_gripper_qpos into a single state feature
    # Only do this if all required keys exist
    features[OBS_STATE] = {
        "dtype": str(state_dtypes["robot0_joint_pos_cos"]),
        "shape": (state_shapes["robot0_joint_pos_cos"][0] + state_shapes["robot0_joint_pos_sin"][0] + state_shapes["robot0_gripper_qpos"][0],),
        "names": ["robot0_joint_pos_cos_7d", "robot0_joint_pos_sin_7d", "robot0_gripper_qpos_2d"],
    }
    
    # Add actions
    features[ACTION] = {
        "dtype": str(properties["action_dtype"]),
        "shape": properties["action_shape"],
        "names": [ACTION],
    }

    # Add other episode-level datasets and groups
    for key, value in properties["other_keys"].items():
        features[key] = {
            "dtype": str(value[1]),
            "shape": value[0],
            "names": [key],
        }

    features["task_clip_embedding"] = {
        "dtype": "float32",
        "shape": (512,),
        "names": ["task_clip_embedding"],
    }
    return features


def load_robocasa_episode(f: h5py.File, ep_name: str, other_keys: Dict[str, tuple] = None) -> Dict[str, Any]:
    """
    Load a single episode from RoboCasa HDF5 file.
    Loads ALL available data including observations, actions, and any other datasets/groups.
    
    Args:
        f: Open HDF5 file
        ep_name: Episode name
        other_keys: Dict of other dataset keys to load (from discover_dataset_properties)
    """
    ep_group = f["data"][ep_name]
    
    # Load observations
    obs_group = ep_group["obs"]
    obs = {}
    for key in obs_group.keys():
        obs[key] = obs_group[key][:]
    
    # Load actions
    actions = ep_group["actions"][:]
    
    # Load other episode-level datasets (like states, policy_mode, actions_abs, etc.)
    other_data = {}
    for key in other_keys.keys():
        other_data[key] = ep_group[key][:]
    
    # Load all episode attributes
    ep_attrs = {}
    for attr_name in ep_group.attrs.keys():
        attr_value = ep_group.attrs[attr_name]
        attr_value = attr_value.decode("utf-8") if isinstance(attr_value, bytes) else attr_value
        ep_attrs[attr_name] = attr_value
    
    # Load episode metadata for language instruction
    ep_meta = {}
    if "ep_meta" in ep_group.attrs:
        ep_meta_str = ep_group.attrs["ep_meta"]
        if isinstance(ep_meta_str, bytes):
            ep_meta_str = ep_meta_str.decode("utf-8")
        if isinstance(ep_meta_str, str):
            ep_meta = json.loads(ep_meta_str)
    
    lang = ep_meta.get("lang", "") if ep_meta else ""
    
    return {
        "obs": obs,
        "actions": actions,
        "lang": lang,
        "ep_meta": ep_meta,
        "other_data": other_data,  # Other episode-level datasets
        "ep_attrs": ep_attrs,  # All episode attributes
    }


def convert_robocasa_to_lerobot(
    dataset_paths: List[str],
    repo_name: str,
    push_to_hub: bool = False,
    n_episodes: int = -1,
    video_codec: str = "libsvtav1",  # Video encoding backend: "libsvtav1" (recommended), "h264", or "hevc"
):
    """
    Convert RoboCasa HDF5 datasets to LeRobot format.
    
    Args:
        dataset_paths: List of paths to RoboCasa HDF5 dataset files
        repo_name: Name for the LeRobot dataset repository
        push_to_hub: Whether to push to Hugging Face Hub
        n_episodes: Number of episodes to convert (if -1, convert all episodes)
    """
    # Discover properties from first dataset
    print(f"Discovering properties from {dataset_paths[0]}...")
    properties = discover_dataset_properties(dataset_paths)
    
    print(f"Discovered properties:")
    print(f"  Image names: {properties['image_names']}")
    print(f"  Image shapes: {properties['image_shapes']}")
    print(f"  State keys: {properties['state_keys']}")
    print(f"  State shapes: {properties['state_shapes']}")
    print(f"  Action shape: {properties['action_shape']}")
    print(f"  FPS (from dataset): {properties['fps']}")
    if properties.get('other_keys'):
        print(f"  Other episode-level datasets: {list(properties['other_keys'].keys())}")
    print(f"  Max dims: {properties['max_dims']}")
    # Use discovered FPS or provided/default
    fps = properties["fps"]
    print(f"  Using FPS: {fps}")
    
    # Create features
    features = create_lerobot_features(properties)
    print(f"\nCreated features: {list(features.keys())}")
    for key, value in features.items():
        print(f"  {key}: dtype={value.get('dtype', 'N/A')}, shape={value.get('shape', 'N/A')}")
    
    # Extract dataset names and group paths by dataset name
    # WARNING! This is highly specific to RoboCasa dataset saving structure.
    # Path format: /path/to/dataset/DATASET_NAME/TIMESTAMP/demo_*.hdf5
    # We extract DATASET_NAME which is the third-to-last path component
    dataset_groups = defaultdict(list)
    for dataset_path in dataset_paths:
        dataset_name = dataset_path.split("/")[-3]
        dataset_groups[dataset_name].append(dataset_path)
    
    print(f"\nGrouped {len(dataset_paths)} paths into {len(dataset_groups)} dataset(s):")
    for dataset_name, paths in dataset_groups.items():
        print(f"  {dataset_name}: {len(paths)} path(s)")
    
    # Process each dataset group
    all_total_episodes = 0
    all_datasets = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    for dataset_name, paths in dataset_groups.items():
        dataset_repo_id = f"{repo_name}-{dataset_name}"
        output_path = HF_LEROBOT_HOME / dataset_repo_id
        
        # Clean up any existing dataset
        if output_path.exists():
            print(f"\nCleaning up existing dataset at {output_path}...")
            shutil.rmtree(output_path)

        # Create LeRobot dataset for this dataset name
        print(f"\nCreating LeRobot dataset '{dataset_repo_id}'...")
        print(f"Video encoding backend: {video_codec}")
        dataset = LeRobotDataset.create(
            repo_id=dataset_repo_id,
            robot_type="PandaOmron",  # Default, in robocasa datasets
            fps=fps,
            features=features,
            image_writer_threads=10,
            image_writer_processes=5,
        )
        
        # Override video encoding method to use specified codec
        # Uses blog post recommended defaults: g=2, crf=30, pix_fmt=yuv420p
        def custom_encode_video(video_key: str, episode_index: int):
            """Custom video encoding with specified codec backend."""
            import tempfile
            import shutil
            temp_path = Path(tempfile.mkdtemp(dir=dataset.root)) / f"{video_key}_{episode_index:03d}.mp4"
            img_dir = dataset._get_image_file_dir(episode_index, video_key)
            encode_video_frames(
                img_dir, 
                temp_path, 
                dataset.fps, 
                vcodec=video_codec,
                pix_fmt="yuv420p",  # Recommended pixel format
                g=2,  # GOP size (recommended for robotics datasets)
                crf=30,  # Quality setting (recommended)
                overwrite=True
            )
            shutil.rmtree(img_dir)
            return temp_path
        
        dataset._encode_temporary_episode_video = custom_encode_video

        # Process all paths for this dataset name
        total_episodes = 0
        ep_meta_list = []
        for dataset_path in paths:
            print(f"\nProcessing {dataset_path}...")
            with h5py.File(dataset_path, "r") as f:
                demos = sorted(list(f["data"].keys()))
                print(f"  Found {len(demos)} episodes")
                
                for ep_name in demos:
                    ep_data = load_robocasa_episode(
                        f, ep_name, 
                        other_keys=properties.get("other_keys"),
                    )
                    ep_meta = ep_data["ep_meta"]
                    lang = ep_meta.get("lang")
                    ep_meta_list.append(ep_data["ep_meta"])
                    task_clip_embedding = get_clip_embedding(lang, tokenizer, model, device)
                    
                    # Get number of timesteps
                    num_steps = len(ep_data["actions"])
                    
                    # Process each timestep
                    for t in range(num_steps):
                        frame_data = {}
                        
                        # Add images using their original names
                        for img_name in properties["image_names"]:
                            frame_data[OBS_IMAGES + "." + img_name] = ep_data["obs"][img_name + "_image"][t]
                        
                        # Add each state key as a separate feature (use ALL keys found in episode)
                        for state_key in properties["state_keys"]:
                            if 'obs/' + state_key in properties["max_dims"]:
                                frame_data[state_key] = np.full(properties["max_dims"][f'obs/{state_key}'], -np.inf)
                                frame_data[state_key][:ep_data["obs"][state_key][t].shape[0]] = ep_data["obs"][state_key][t]
                            else:
                                frame_data[state_key] = ep_data["obs"][state_key][t]
                        for key in properties["other_keys"].keys():
                            data = np.array(ep_data["other_data"][key][t])
                            if data.shape == ():
                                data = np.array([data])
                            if key in properties["max_dims"]:
                                # pad it with -np.inf
                                frame_data[key] = np.full(properties["max_dims"][key], -np.inf)
                                frame_data[key][:data.shape[0]] = data
                            else:
                                frame_data[key] = data

                        # Combine robot state keys into OBS_STATE
                        arm_state = np.concatenate([
                            ep_data["obs"]["robot0_joint_pos_cos"][t], 
                            ep_data["obs"]["robot0_joint_pos_sin"][t]
                        ], axis=0)
                        gripper_state = ep_data["obs"]["robot0_gripper_qpos"][t]
                        frame_data[OBS_STATE] = np.concatenate([arm_state, gripper_state], axis=0)
                        
                        # Add actions
                        frame_data[ACTION] = ep_data["actions"][t]
                        
                        # Add language instruction
                        lang = ep_data["lang"] # only language instruction is available in general
                        frame_data["task"] = lang
                        frame_data["task_clip_embedding"] = torch.from_numpy(task_clip_embedding).float()
                        
                        # this is v3 format of saving data
                        dataset.add_frame(frame_data)

                    
                    # save the ep_meta for the episode in the dataset
                    dataset.save_episode()
                    total_episodes += 1
                    all_total_episodes += 1
                    
                    if n_episodes > 0 and all_total_episodes >= n_episodes:
                        break
                
                if n_episodes > 0 and all_total_episodes >= n_episodes:
                    break
        
        dataset.finalize()
        # add the ep_meta for the dataset in the info.json file
        with open(output_path / "meta/episodes/ep_metas.json", "w") as f:
            json.dump(ep_meta_list, f)
        all_datasets.append((dataset, dataset_repo_id))
        print(f"\n✅ Converted {total_episodes} episodes for dataset '{dataset_name}'!")
        print(f"Dataset saved to: {output_path}")
    
    print(f"\n✅ Total: Converted {all_total_episodes} episodes across {len(dataset_groups)} dataset(s)!")

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        print("\nPushing all datasets to Hugging Face Hub...")
        for dataset, dataset_repo_id in all_datasets:
            print(f"Pushing {dataset_repo_id}...")
            dataset.push_to_hub(
                tags=["robocasa", "panda", "kitchen"],
                private=True,
                push_videos=True,
                license="apache-2.0",
            )
            print(f"✅ Pushed {dataset_repo_id}!")


def main(
    dataset_path: Union[str, List[str]],
    repo_name: str,
    *,
    push_to_hub: bool = False,
    n_episodes: int = -1,
    video_codec: str = "libsvtav1",  # Video encoding backend: "libsvtav1" (recommended), "h264", or "hevc"
):
    """
    Convert RoboCasa HDF5 datasets to LeRobot format.
    
    Args:
        dataset_path: Path(s) to RoboCasa HDF5 dataset file(s). Use the dataset with images in it (see robocasa for details).
        repo_name: Name for the LeRobot dataset repository (e.g., "your_hf_username/robocasa_dataset")
        push_to_hub: Whether to push to Hugging Face Hub
        n_episodes: Number of episodes to convert (if -1, convert all episodes)
    """
    if isinstance(dataset_path, str):
        dataset_paths = [dataset_path]
    else:
        dataset_paths = list(dataset_path)
    if repo_name.endswith("/"):
        repo_name = repo_name[:-1]
    
    convert_robocasa_to_lerobot(
        dataset_paths=dataset_paths,
        repo_name=repo_name,
        push_to_hub=push_to_hub,
        n_episodes=n_episodes,
        video_codec=video_codec,
    )

if __name__ == "__main__":
    tyro.cli(main)
