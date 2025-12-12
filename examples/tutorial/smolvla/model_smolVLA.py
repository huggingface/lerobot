import torch
import os
import sys
import time
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# print(sys.path)
# # Assuming lerobot is installed
# from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
# #from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from LTA_Pool.datasets import BatchData, build_dataloader
#from LTA_Pool.datasets.nuscenes.dataloader import ImageLoader





def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "lerobot/smolvla_base"

    print(f"🔄 Loading model: {model_id}...")
    
    # 1. Load the default config from the Hub
    model_id = "lerobot/smolvla_base"
    config = SmolVLAConfig.from_pretrained(model_id)

    # 2. Inject your custom input features manually
    # Note: We must use the PolicyFeature class, not raw dictionaries
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.camera3": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    }
    # config.max_action_dim = 2
    # config.max_state_dim = 8
    model = SmolVLAPolicy.from_pretrained(model_id, config=config)
    model = model.to(device)
    
    # --- Configuration Parameters ---
    B = 2                 
    C = 3
    # Note: input_features shape is (256, 256), but model resizes to (512, 512) internally
    # defined by config.resize_imgs_with_padding
    H, W = 512,512       
    
    # Check config for dimensions
    MAX_STATE_DIM = getattr(model.config, "max_state_dim", 10)
    CHUNK_SIZE = getattr(model.config, "chunk_size", 50)
    # Check config for action dim, default to 32 based on your inspection output
    MAX_ACTION_DIM = getattr(model.config, "max_action_dim", 32)
    TOKENIZER_MAX_LEN = getattr(model.config, "tokenizer_max_length", 48)
    RESIZE_DIMS = getattr(model.config, "resize_imgs_with_padding", (H, W))

    print(f"✅ Config: Action Dim={MAX_ACTION_DIM}, Chunk={CHUNK_SIZE}, Input Img={H}x{W}")
    print(f"   Internal Resize Target: {RESIZE_DIMS}")
    print(f"MAX_STATE_DIM: {MAX_STATE_DIM}")
    # --- 1. Create Input Tensors ---
    dummy_image = torch.randn(B, C, H, W, device=device)
    dummy_state = torch.randn(B, MAX_STATE_DIM, device=device)
    dummy_input_ids = torch.randint(0, 1000, (B, TOKENIZER_MAX_LEN), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones(B, TOKENIZER_MAX_LEN, dtype=torch.bool, device=device)
    dummy_action = torch.randn(B, CHUNK_SIZE, MAX_ACTION_DIM, device=device)
    breakpoint()
    # --- 2. Assemble Input Dictionary ---
    # UPDATE: We must use the 3 specific camera keys found in inspection.
    model_input = {
        # Camera Inputs (Found 3 in inspection)
        "observation.images.camera1": dummy_image,
        "observation.images.camera2": dummy_image,
        "observation.images.camera3": dummy_image,
        
        # Other Inputs (Corrected keys to actual strings model expects)
        "observation.state": dummy_state,
        
        # UPDATE: Corrected Language Keys based on KeyError
        # The model code expects these exact strings
        "observation.language.tokens": dummy_input_ids,
        "observation.language.attention_mask": dummy_attention_mask,
        
        # Action (Required for training forward pass)
        "action": dummy_action
    }
    print("======================================")
    print(f"{model_input['observation.images.camera1'].shape}")
    print(f"{model_input['observation.images.camera2'].shape}")
    print(f"{model_input['observation.images.camera3'].shape}")
    print(f"Language Tokens: {model_input['observation.language.tokens'].shape}")
    print(f"{model_input['observation.language.attention_mask'].shape}")
    print(f"State : {model_input['observation.state'].shape}")
    print(f"Actions Shape : {model_input['action'].shape}")
    
    print("======================================")
    
    print("\n--- 1. Simulating Training Forward Pass (Loss Calculation) ---")
    
    # CRITICAL: Pass 'batch=model_input'. Do not use **model_input.
    loss, loss_dict, v_t = model(batch=model_input)
    
    print(f"✅ Forward pass successful.")
    print(f"   Loss value: {loss.item():.4f}")
    print(f"   Model Output shape: {v_t.shape}")

    print("\n--- 2. Simulating Inference (Predict Action Chunk) ---")
    
    # Remove 'action' ground truth for inference
    inference_input = {k: v for k, v in model_input.items() if k != "action"}
    
    with torch.no_grad():
        predicted_actions = model.predict_action_chunk(batch=inference_input)
    
    print(f"✅ Inference successful.")
    print(f"   Predicted Shape: {predicted_actions.shape}")
    print(f"   Expected: [{B}, {CHUNK_SIZE}, {MAX_ACTION_DIM}]")


def dataset_class():
    dataset_configs = [
        {
            "name": "nuscenes",
            "kwargs": {
                "load_images": True,
                "num_past_images": 6,  # max 6
                "image_res": (224, 224),  # max (512, 512)
            },
        }
    ]
    dataloader = build_dataloader(
        dataset_configs=dataset_configs,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
    )
    print(f"Total samples: {len(dataloader.dataset)}\n")

    start = time.time()

    batch: BatchData
    for i, batch in enumerate(dataloader):
        if i == 0:
            print(f"{batch.history.cam_front.shape=}")
            print(f"{batch.gt_traj.shape=}")
            print(f"{batch.ego_curr_state.shape=}")
            print(f"{batch.prompt.shape=}")
            print(f"{batch.history.trajectory.shape=}")
            print(batch.prompt2)
            print(len(batch.captions1), batch.captions1[0])
            print(len(batch.captions2), batch.captions1[1])
            breakpoint()

        if i == 5:
            break

    print(f"\nTime for {i} batches: {time.time() - start:.2f}s")
    
if __name__ == "__main__":
    main()
    #dataset_class()
    
    
class SmolVLASyntheticDataset(Dataset):
    """
    A synthetic dataset that generates random inputs for SmolVLA testing.
    """
    def __init__(self, length=100, config=None):
        self.length = length
        
        # Default dimensions based on your snippet/config
        self.c = 3
        self.h = 512
        self.w = 512
        self.state_dim = 32
        self.action_dim = 2
        self.chunk_size = 50
        self.tokenizer_max_len = 48
        
        # If config is passed, override defaults
        if config:
            self.state_dim = getattr(config, "max_state_dim", self.state_dim)
            self.action_dim = getattr(config, "max_action_dim", self.action_dim)
            self.chunk_size = getattr(config, "chunk_size", self.chunk_size)
            self.tokenizer_max_len = getattr(config, "tokenizer_max_length", self.tokenizer_max_len)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Generates ONE single sample. 
        Note: No 'B' (Batch) dimension here. The DataLoader adds it.
        """
        
        # 1. Images: Use torch.rand for [0, 1] range (better for normalization)
        # Shape: (3, 512, 512)
        dummy_image = torch.rand(self.c, self.h, self.w)

        # 2. State: Standard Normal
        # Shape: (32)
        dummy_state = torch.randn(self.state_dim)

        # 3. Language Tokens: Random Integer IDs
        # Shape: (48)
        dummy_input_ids = torch.randint(0, 1000, (self.tokenizer_max_len,), dtype=torch.long)
        
        # 4. Attention Mask: All ones (attend to everything)
        # Shape: (48)
        dummy_attention_mask = torch.ones(self.tokenizer_max_len, dtype=torch.bool)

        # 5. Action Chunk: Standard Normal
        # Shape: (50, 32)
        dummy_action = torch.randn(self.chunk_size, self.action_dim)

        # 6. Assemble Dictionary with EXACT keys
        sample = {
            "observation.images.camera1": dummy_image,
            # "observation.images.camera2": dummy_image,
            # "observation.images.camera3": dummy_image,
            
            "observation.state": dummy_state,
            
            "observation.language.tokens": dummy_input_ids,
            "observation.language.attention_mask": dummy_attention_mask,
            
            "action": dummy_action
        }
        
        return sample