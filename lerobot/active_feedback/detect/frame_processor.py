import torch
import numpy as np
import cv2

class FrameProcessor:
    """
    Handles extraction and preprocessing of frames from LeRobotDataset.
    
    This class provides utilities to extract frames from episodes, convert between
    tensor and numpy formats, and save frames to disk if needed.
    """
    
    def __init__(self, dataset, debug=False):
        """
        Initialize the frame processor.
        
        Args:
            dataset (LeRobotDataset): The loaded robot dataset
            debug (bool): Whether to enable debug mode with additional logging
        """
        self.dataset = dataset
        self.debug = debug
        
    def get_episode_frames(self, episode_index, camera_key=None):
        """
        Extract all frames from a specific episode.
        
        Args:
            episode_index (int): Index of the episode
            camera_key (str, optional): Camera key to use. If None, uses the first camera.
            
        Returns:
            list: List of frames as tensors
        """
        if camera_key is None:
            camera_key = self.dataset.meta.camera_keys[0]
            
        from_idx = self.dataset.episode_data_index["from"][episode_index].item()
        to_idx = self.dataset.episode_data_index["to"][episode_index].item()
        
        frames = [self.dataset[idx][camera_key] for idx in range(from_idx, to_idx)]
        
        if self.debug:
            print(f"Extracted {len(frames)} frames from episode {episode_index}")
            
        return frames
    
    @staticmethod
    def tensor_to_numpy(tensor):
        """
        Convert a PyTorch tensor to a numpy array suitable for OpenCV.
        
        Args:
            tensor (torch.Tensor): Input tensor in (C, H, W) format
            
        Returns:
            numpy.ndarray: Array in (H, W, C) format with values in [0, 255] range
        """
        # Handle different tensor shapes based on delta_timestamps
        if len(tensor.shape) == 4:  # (T, C, H, W)
            # Take the most recent frame (last in sequence)
            tensor = tensor[-1]
            
        # Convert from (C, H, W) to (H, W, C)
        np_img = tensor.permute(1, 2, 0).cpu().numpy()
        
        # Convert to uint8 if needed
        if np_img.dtype != np.uint8:
            if np_img.max() <= 1.0:
                np_img = (np_img * 255).astype(np.uint8)
            else:
                np_img = np_img.astype(np.uint8)
                
        # Convert from RGB to BGR for OpenCV
        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        
        return np_img
    
    def save_frame(self, frame, path):
        """
        Save a frame to disk.
        
        Args:
            frame (torch.Tensor or numpy.ndarray): Frame to save
            path (str): Path to save the frame
            
        Returns:
            bool: True if successful
        """
        if isinstance(frame, torch.Tensor):
            frame = self.tensor_to_numpy(frame)
            
        return cv2.imwrite(path, frame)