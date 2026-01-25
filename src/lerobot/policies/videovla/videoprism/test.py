import torch
import numpy as np
from torchcodec.decoders import VideoDecoder

from lerobot.policies.videovla.videoprism import VideoPrismVideoProcessor
from lerobot.policies.videovla.videoprism import VideoPrismVisionModel
processor = VideoPrismVideoProcessor.from_pretrained(
    "MHRDYN7/videoprism-base-f16r288"
)

model = VideoPrismVisionModel.from_pretrained(
    "MHRDYN7/videoprism-base-f16r288",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
)

video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"

vr = VideoDecoder(video_url)
frame_idx = np.arange(0, 64)
video = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W

video = processor(video, return_tensors="pt")
video = {k: v.to(model.device, model.dtype) for k, v in video.items()}
outputs = model(**video)
encoder_outputs = outputs.last_hidden_state
print(encoder_outputs.shape) # 

import time
import torch

# warmup
for _ in range(10):
    _ = model(**video)

times = []
for _ in range(50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    _ = model(**video)

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)

print(f"Mean: {1000*sum(times)/len(times):.2f} ms")
print(f"Min : {1000*min(times):.2f} ms")
print(f"Max : {1000*max(times):.2f} ms")
