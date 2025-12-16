import numpy as np
from transformers import AutoProcessor
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

delta_timestamps = {'action': [0.0, 0.03333333333333333, 0.06666666666666667, 0.1, 0.13333333333333333, 0.16666666666666666, 0.2, 0.23333333333333334, 0.26666666666666666, 0.3, 0.3333333333333333, 0.36666666666666664, 0.4, 0.43333333333333335, 0.4666666666666667, 0.5, 0.5333333333333333, 0.5666666666666667, 0.6, 0.6333333333333333, 0.6666666666666666, 0.7, 0.7333333333333333, 0.7666666666666667, 0.8, 0.8333333333333334, 0.8666666666666667, 0.9, 0.9333333333333333, 0.9666666666666667, 1.0, 1.0333333333333334, 1.0666666666666667, 1.1, 1.1333333333333333, 1.1666666666666667, 1.2, 1.2333333333333334, 1.2666666666666666, 1.3, 1.3333333333333333, 1.3666666666666667, 1.4, 1.4333333333333333, 1.4666666666666666, 1.5, 1.5333333333333334, 1.5666666666666667, 1.6, 1.6333333333333333]}
dataset = LeRobotDataset(repo_id="local", root="/fsx/jade_choghari/outputs/pgen_annotations1", delta_timestamps=delta_timestamps)

dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=4,
        shuffle=True,
)

batch = next(iter(dataloader))

# Load the tokenizer from the Hugging Face hub
tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

# Tokenize & decode action chunks (we use dummy data here)
action_data = batch["action"]    # one batch of action chunks
tokens = tokenizer(action_data)              # tokens = list[int]
decoded_actions = tokenizer.decode(tokens)
print("tokenized actions: ", tokens)
