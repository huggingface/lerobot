import numpy as np
import torch
import cv2
from typing import List, Dict, Tuple, Optional, Union
import tempfile
from pathlib import Path
import subprocess

class ACTPolicyWithReward:
    """
    Wrapper for ACTPolicy that captures reward values from the third return value of select_action.
    """
    
    def __init__(self, policy, recording_enabled: bool = True):
        """
        Initialize the wrapper with an ACTPolicy.
        
        Args:
            policy: An instance of ACTPolicy
            recording_enabled: Whether to record rewards and images for video generation
        """
        self.policy = policy
        self.config = policy.config
        self.recording_enabled = recording_enabled
        
        # Recording data
        self.recorded_rewards = []
        self.recorded_images = []
        self.step_count = 0
        
        # For storing the last processed data
        self.last_observation = None
        self.last_reward = None
        
    def select_action(self, observation: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        """
        Extends policy.select_action to capture reward values.
        
        Args:
            observation: Dictionary of observations
            
        Returns:
            action: The predicted action tensor
            reward: The reward value (float between 0 and 1)
        """
        # Store the observation for later use
        self.last_observation = observation.copy()
        
        # Extract images for recording
        if self.recording_enabled:
            images = self._extract_images(observation)
            self.recorded_images.append(images)
        
        # Call the original policy's select_action
        with torch.inference_mode():
            action, reward = self.policy.select_action(observation, force_model_run=True)
        
        # Extract reward value (should be a float between 0 and 1)
        if isinstance(reward, torch.Tensor):
            reward_value = float(reward.item())
        else:
            reward_value = float(reward)
        
        # Clamp reward to [0, 1] range
        reward_value = max(0.0, min(1.0, reward_value))
        
        self.last_reward = reward_value
        
        # Record reward for video generation if enabled
        if self.recording_enabled:
            self.recorded_rewards.append({
                'step': self.step_count,
                'reward': reward_value
            })
        
        self.step_count += 1
        
        return action, reward_value
    
    def _extract_images(self, observation: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Extract image tensors from observation dictionary"""
        images = []
        for key in observation:
            if "image" in key:
                images.append(observation[key].clone())
        return images
    
    def get_reward_statistics(self) -> Dict[str, float]:
        """Get statistics about recorded rewards"""
        if not self.recorded_rewards:
            return {}
        
        rewards = [r['reward'] for r in self.recorded_rewards]
        
        return {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'max_reward': float(np.max(rewards)),
            'min_reward': float(np.min(rewards)),
            'final_reward': rewards[-1] if rewards else 0.0,
            'total_steps': len(rewards)
        }
    
    def clear_recorded_data(self):
        """Clear recorded data to free memory"""
        self.recorded_rewards.clear()
        self.recorded_images.clear()
        self.step_count = 0
        print("Reward recording data cleared.")
    
    def get_current_reward(self) -> Optional[float]:
        """Get reward from the last step"""
        return self.last_reward
    
    # Forward other methods to the original policy
    def __getattr__(self, name):
        if name not in self.__dict__:
            return getattr(self.policy, name)
        return self.__dict__[name]


def create_reward_visualization_video(
    images_list: List[List[torch.Tensor]], 
    reward_data: List[Dict], 
    output_path: str = "reward_visualization.mp4",
    fps: int = 20,
    image_size: Tuple[int, int] = (640, 480),
    graph_height: int = 200
) -> str:
    """
    Create a video showing images with reward line graph below.
    
    Args:
        images_list: List of image tensors for each step
        reward_data: List of reward dictionaries from ACTPolicyWithReward
        output_path: Path for output video
        fps: Frames per second
        image_size: Size for resized images
        graph_height: Height of the reward graph section
        
    Returns:
        Path to created video
    """
    if not images_list or not reward_data:
        print("No images or reward data provided for video generation")
        return None
    
    print(f"Creating reward visualization video with {len(images_list)} frames...")
    
    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Generate frames
        for step_idx, (images, reward_info) in enumerate(zip(images_list, reward_data)):
            frame_path = temp_path / f"frame_{step_idx:06d}.png"
            create_reward_frame(images, reward_info, reward_data, frame_path, image_size, graph_height)
        
        # Create video using ffmpeg
        create_video_from_frames(temp_path, output_path, fps)
    
    print(f"Reward visualization video saved to: {output_path}")
    return output_path


def create_reward_frame(
    images: List[torch.Tensor], 
    current_reward_data: Dict,
    all_reward_data: List[Dict],
    output_path: Path,
    image_size: Tuple[int, int],
    graph_height: int
):
    """Create a single reward visualization frame with images and reward graph"""
    # Configuration
    image_width, image_height = image_size
    margin = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    
    # Get valid images
    valid_images = [img for img in images if img is not None]
    num_images = len(valid_images)
    
    # Calculate canvas dimensions
    if num_images == 0:
        canvas_width = 800
        images_height = image_height
    else:
        canvas_width = num_images * image_width + (num_images + 1) * margin
        images_height = image_height
    
    total_height = images_height + graph_height + 3 * margin
    
    # Create canvas
    canvas = np.zeros((total_height, canvas_width, 3), dtype=np.uint8)
    canvas.fill(40)  # Dark gray background
    
    # Process and place images
    y_img = margin
    for i, img in enumerate(valid_images):
        x_img = margin + i * (image_width + margin)
        
        # Convert tensor to opencv format
        if img.dim() == 4:  # (B,C,H,W)
            img = img.squeeze(0)
        
        # Handle different tensor formats
        if img.dim() == 3:
            if img.shape[0] == 3:  # (C,H,W)
                img_np = img.permute(1, 2, 0).cpu().numpy()
            elif img.shape[2] == 3:  # (H,W,C)
                img_np = img.cpu().numpy()
            else:
                print(f"Warning: Unexpected image shape {img.shape}, skipping image {i}")
                continue
        else:
            print(f"Warning: Unexpected image dimensions {img.dim()}, skipping image {i}")
            continue
        
        # Ensure we have 3 channels
        if img_np.shape[2] != 3:
            print(f"Warning: Image has {img_np.shape[2]} channels, expected 3, skipping image {i}")
            continue
        
        # Normalize to 0-255
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Resize image
        img_resized = cv2.resize(img_np, (image_width, image_height))
        
        # Place image on canvas
        if img_resized.shape == (image_height, image_width, 3):
            canvas[y_img:y_img + image_height, x_img:x_img + image_width] = img_resized
            
            # Add camera label
            label = f'Camera {i + 1}'
            label_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            label_x = x_img + (image_width - label_size[0]) // 2
            label_y = y_img - 5
            cv2.putText(canvas, label, (label_x, label_y), font, font_scale, (255, 255, 255), font_thickness)
    
    # Create reward graph
    graph_y = images_height + 2 * margin
    graph_width = canvas_width - 2 * margin
    graph_area_height = graph_height - 2 * margin
    
    # Graph background
    cv2.rectangle(canvas, (margin, graph_y), (canvas_width - margin, graph_y + graph_height), (60, 60, 60), -1)
    
    # Draw reward line graph
    current_step = current_reward_data['step']
    current_reward = current_reward_data['reward']
    
    # Get all rewards up to current step
    rewards_so_far = [r['reward'] for r in all_reward_data[:current_step + 1]]
    steps_so_far = [r['step'] for r in all_reward_data[:current_step + 1]]
    
    if len(rewards_so_far) > 1:
        # Calculate graph coordinates
        max_steps = max(50, len(all_reward_data))  # Show at least 50 steps worth of space
        
        # Draw graph lines and points
        for i in range(1, len(rewards_so_far)):
            # Calculate positions
            x1 = margin + int((steps_so_far[i-1] / max_steps) * graph_width)
            y1 = graph_y + margin + int((1 - rewards_so_far[i-1]) * graph_area_height)
            x2 = margin + int((steps_so_far[i] / max_steps) * graph_width)
            y2 = graph_y + margin + int((1 - rewards_so_far[i]) * graph_area_height)
            
            # Draw line segment
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw point at current position
            if i == len(rewards_so_far) - 1:
                cv2.circle(canvas, (x2, y2), 4, (0, 255, 255), -1)
    
    # Draw graph borders and labels
    cv2.rectangle(canvas, (margin, graph_y + margin), (canvas_width - margin, graph_y + graph_height - margin), (255, 255, 255), 1)
    
    # Y-axis labels (reward values)
    for i in range(5):  # 0, 0.25, 0.5, 0.75, 1.0
        reward_val = i * 0.25
        y_pos = graph_y + margin + int((1 - reward_val) * graph_area_height)
        cv2.putText(canvas, f'{reward_val:.2f}', (5, y_pos), font, 0.4, (255, 255, 255), 1)
        # Draw horizontal grid line
        cv2.line(canvas, (margin, y_pos), (canvas_width - margin, y_pos), (100, 100, 100), 1)
    
    # Current reward display
    reward_text = f'Current Reward: {current_reward:.3f}'
    text_size = cv2.getTextSize(reward_text, font, font_scale * 1.2, font_thickness + 1)[0]
    text_x = (canvas_width - text_size[0]) // 2
    text_y = graph_y + graph_height - 10
    cv2.putText(canvas, reward_text, (text_x, text_y), font, font_scale * 1.2, (255, 255, 255), font_thickness + 1)
    
    # Step counter
    step_text = f'Step {current_step}'
    cv2.putText(canvas, step_text, (margin, graph_y - 5), font, font_scale, (255, 255, 255), font_thickness)
    
    # Save frame
    cv2.imwrite(str(output_path), canvas)


def create_video_from_frames(frames_dir: Path, output_path: str, fps: int):
    """Use ffmpeg to create video from frames"""
    cmd = [
        'ffmpeg', '-y',  # -y to overwrite output file
        '-framerate', str(fps),
        '-i', str(frames_dir / 'frame_%06d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',  # Good quality
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        print(f"FFmpeg output: {e.stdout.decode()}")
        print(f"FFmpeg errors: {e.stderr.decode()}")
        raise
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg to create videos.")
