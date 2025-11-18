# LeRobot Embedding Generation Script

Generate embeddings for LeRobot datasets to make them more lightweight and efficient for training.

## Overview

This script processes v3.0 LeRobot datasets and adds pre-computed embeddings for:

- **Task embeddings**: Language command embeddings using MiniLM
- **Image embeddings**: Frame embeddings using DinoV2

The resulting dataset can be used more efficiently during training by loading pre-computed embeddings instead of running encoders on-the-fly.

## Supported Encoders

### Image Encoders (DinoV2)

DinoV2 is a self-supervised vision transformer that produces high-quality image embeddings:

- **`dinov2_vits14`**: ViT-S/14 (384-dim) - Fastest, smaller model
- **`dinov2_vitb14`**: ViT-B/14 (768-dim) - **Recommended** - Good balance
- **`dinov2_vitl14`**: ViT-L/14 (1024-dim) - Best quality, slower

### Language Encoders (MiniLM)

MiniLM is a lightweight sentence transformer model:

- **`minilm-l6`**: MiniLM-L6-v2 (384-dim) - Faster
- **`minilm-l12`**: MiniLM-L12-v2 (384-dim) - **Recommended** - Better quality

## Usage

### Basic Command

```bash
python src/lerobot/datasets/generating_embeddings/generate_embeddings.py \
    --repo-id lerobot/utokyo_xarm_bimanual \
    --output-repo-id your-username/utokyo_xarm_bimanual_embeddings \
    --image-encoder dinov2_vitb14 \
    --language-encoder minilm-l12 \
    --push-to-hub
```

### Lightweight Version (No Videos)

Removes video files to significantly reduce storage:

```bash
python src/lerobot/datasets/generating_embeddings/generate_embeddings.py \
    --repo-id lerobot/utokyo_xarm_bimanual \
    --output-repo-id your-username/utokyo_xarm_bimanual_lightweight \
    --image-encoder dinov2_vitb14 \
    --language-encoder minilm-l12 \
    --remove-videos \
    --push-to-hub
```

## Output

The script adds new features to your dataset:

### New Features

1. **`task_embedding`**: Language embedding for each frame
   - Shape: `[384]` (MiniLM)
   - One embedding per frame based on its task

2. **`{camera_key}_embedding`**: Image embedding for each camera view
   - Shape: `[384]`, `[768]`, or `[1024]` depending on DinoV2 model
   - Examples: `observation.images.top_embedding`, `observation.images.wrist_embedding`

### Using Embeddings in Training

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load dataset with embeddings
dataset = LeRobotDataset("your-username/utokyo_xarm_bimanual_embeddings")

# Access embeddings
item = dataset[0]
task_emb = item["task_embedding"]  # Shape: [384]
img_emb = item["observation.images.top_embedding"]  # Shape: [768]

# Use in your policy
# Instead of running encoders during training, use pre-computed embeddings
```

## Extending with New Encoders

The script is designed to be easily extensible. To add a new encoder:

### 1. Create Encoder Class

```python
class MyCustomImageEncoder(ImageEncoder):
    """Your custom image encoder."""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        # Load your model
        self.model = load_my_model()
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        """Encode a batch of images."""
        # Your encoding logic here
        embeddings = []
        for img in images:
            emb = self.model(img)
            embeddings.append(emb)
        return np.array(embeddings)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return 512  # Your embedding dimension
```

### 2. Add to Factory Function

```python
def get_image_encoder(encoder_name: str, device: str = "cuda") -> ImageEncoder:
    encoders = {
        "dinov2_vits14": lambda: DinoV2Encoder(model_name="dinov2_vits14", device=device),
        "dinov2_vitb14": lambda: DinoV2Encoder(model_name="dinov2_vitb14", device=device),
        "dinov2_vitl14": lambda: DinoV2Encoder(model_name="dinov2_vitl14", device=device),
        # Add your encoder
        "my_custom": lambda: MyCustomImageEncoder(device=device),
    }
    # ... rest of function
```

## Validating Embeddings

After generating embeddings, you can validate them using `validate_embeddings.py`:

```bash
python src/lerobot/datasets/generating_embeddings/validate_embeddings.py \
    --original-repo-id lerobot/utokyo_xarm_bimanual \
    --embeddings-repo-id pepijn223/utokyo_xarm_bimanual_embeddings \
    --image-encoder dinov2_vitb14 \
    --language-encoder minilm-l12 \
    --num-samples 20
```
