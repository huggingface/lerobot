from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi
from safetensors.torch import save_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.videovla.modeling_pi05 import PI05VideoPolicy

# Load config and enable video encoder
config = PreTrainedConfig.from_pretrained("/raid/jade/models/pi05-video")
config.use_video_encoder = True
config.device = "cuda"

# Load model with video encoder enabled
policy = PI05VideoPolicy.from_pretrained("/raid/jade/models/pi05-video", config=config)

policy.push_to_hub("lerobot/pi05-video-1", private=True)

print("done")