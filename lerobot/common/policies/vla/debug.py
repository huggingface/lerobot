import torch
from configuration_vla import Qwen2VLConfig
from modeling_vision import Qwen2VisionTransformerPretrainedModel
from modeling_vla import VLAPolicy


def test_vla_policy():
    # Define the model configuration
    config = Qwen2VLConfig(
        vocab_size=30522,  # Token vocabulary size
        hidden_size=768,  # Hidden size for the model
        num_hidden_layers=2,  # Number of layers in the transformer
        input_shapes={
            "observation.state": [128],  # Observation state shape
        },
        output_shapes={
            "action": [64],  # Action output shape
        },
    )

    # Initialize the VLAPolicy
    vla_policy = VLAPolicy(config)

    # Create a batch of random input data for testing
    batch = {
        "input_ids": torch.randint(
            0, config.vocab_size, (1, 10)
        ),  # Random tokenized input (batch_size=1, seq_len=10)
        "attention_mask": torch.ones((1, 12), dtype=torch.long),  # Attention mask
        "observation.state": torch.randn(1, 128),  # Random observation state (batch_size=1, state_dim=128)
        "action": torch.randn(1, 64),  # Random ground-truth action (batch_size=1, action_dim=64)
    }

    # Perform a forward pass for training (to calculate loss)
    output = vla_policy(batch)
    print("Output during training (loss):", output)

    # Perform action selection (no loss calculated, just action prediction)
    with torch.no_grad():
        predicted_action = vla_policy.select_action(batch)
        print("Predicted Action:", predicted_action)

    _ = Qwen2VisionTransformerPretrainedModel._from_config(
        config.vision_config, attn_implementation=config._attn_implementation
    )


# Run the test function
if __name__ == "__main__":
    test_vla_policy()
"""
def test_model_forward_pass():
    # Define the model configuration
    config = Qwen2VLConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=2,
        input_shapes={
            "observation.state": [128],  # Example observation state shape
        },
        output_shapes={
            "action": [64],  # Example action shape
        },
    )
    # Initialize the VLA model
    model = VLA(config)

    # Generate random input data
    batch = {
        "observation.state": torch.randn(1, 128),  # Batch of size 1, observation state with 128 features
        "action": torch.randn(1, 64),  # Batch of size 1, action with 64 features
    }

    input_ids = torch.randint(0, config.vocab_size, (1, 10))  # Random tokenized input, seq length = 10
    attention_mask = torch.ones((1, 12), dtype=torch.long)  # Attention mask for the sequence and additional tokens

    # Perform forward pass with the model
    with torch.no_grad():  # No gradient needed for testing
        output = model(
            batch=batch,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # Check if output has the correct shape
    assert output.shape == (1, 64), f"Unexpected output shape: {output.shape}"

    print("VLA model forward pass successful with output shape:", output.shape)

if __name__ == "__main__":
    test_model_forward_pass()

"""
"""
    # Initialize the model
    model = Qwen2VLModel(config)

    # Generate random input data
    batch = {
        "observation.state": torch.randn(1, 128),  # Batch of size 1, observation state with 128 features
        "action": torch.randn(1, 64),  # Batch of size 1, action with 64 features
    }

    input_ids = torch.randint(0, config.vocab_size, (1, 10))  # Random tokenized input, seq length = 10
    attention_mask = torch.ones((1, 12), dtype=torch.long)  # Attention mask

    # Perform forward pass with the model
    with torch.no_grad():  # No gradient needed for testing
        output = model(
            batch=batch,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict = True,
            output_hidden_states = True,
        )
    breakpoint()
    # Check if output has the correct shape
    assert output.last_hidden_state.shape == (1, 12, config.hidden_size), \
        f"Unexpected output shape: {output.last_hidden_state.shape}"

    print("Model forward pass successful with output shape:", output.last_hidden_state.shape)

if __name__ == "__main__":
    test_model_forward_pass()
"""
