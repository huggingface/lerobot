import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("2toINF/X-VLA-WidowX", trust_remote_code=True)

processor = AutoProcessor.from_pretrained("2toINF/X-VLA-WidowX", trust_remote_code=True)


# append 3 random image to a list
def make_random_pil_images(num_images=3, H=480, W=640):
    images = []
    for _ in range(num_images):
        # Random RGB image
        arr = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        images.append(img)
    return images


# Example:
images = make_random_pil_images()
language_instruction = "This is a random image"
# Multimodal preprocessing by processor
inputs = processor(images, language_instruction)
if not {"input_ids", "image_input", "image_mask"}.issubset(inputs):
    raise ValueError("Processor did not return the expected keys.")

proprio = torch.randn(1, 20)
domain_id = torch.tensor([0], dtype=torch.long)

# Align to model's device/dtype
device = model.device
dtype = next(model.parameters()).dtype


def to_model(t: torch.Tensor) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    # cast floats to model dtype, keep integral/bool as-is
    return t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device=device)


inputs = {k: to_model(v) for k, v in inputs.items()}
inputs.update(
    {
        "proprio": to_model(proprio),
        "domain_id": domain_id.to(device),
    }
)

# Inference

action = model.generate_actions(**inputs, steps=10).squeeze(0).float().cpu().numpy()

breakpoint()
