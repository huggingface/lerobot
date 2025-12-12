import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def inspect_config():
    model_id = "lerobot/smolvla_base"
    print(f"🔄 Loading model configuration for: {model_id}...")
    
    try:
        # Load the policy (this pulls config.json and weights)
        model = SmolVLAPolicy.from_pretrained(model_id)
        config = model.config
        
        print("\n" + "="*40)
        print(" 🧐 MODEL CONFIGURATION INSPECTION")
        print("="*40)

        # 1. Inspect Image Keys
        # In LeRobot, config.image_features returns a dict of visual inputs
        print("\n📸 [Image Inputs]")
        if hasattr(config, "image_features") and config.image_features:
            print(f"   Found {len(config.image_features)} expected camera input(s):")
            for key, feature in config.image_features.items():
                print(f"    - Key: '{key}'")
                print(f"      Shape: {feature.shape}")
        else:
            print("   ⚠️ No specific 'image_features' found in config. This might be an error or a non-visual policy.")

        # 2. Inspect State/Action Dimensions
        print("\n🤖 [Dimensions]")
        state_dim = getattr(config, "max_state_dim", "Unknown")
        action_dim = getattr(config, "max_action_dim", "Unknown")
        chunk_size = getattr(config, "chunk_size", "Unknown")
        
        print(f"   - Max State Dimension:  {state_dim}")
        print(f"   - Max Action Dimension: {action_dim}")
        print(f"   - Chunk Size:           {chunk_size}")

        return list(config.image_features.keys()) if hasattr(config, "image_features") else []

    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return []

if __name__ == "__main__":
    inspect_config()