import torch
from configuration_vla import Qwen2VLConfig
from modeling_vla import Qwen2VLModel  

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
        )
    breakpoint()
    # Check if output has the correct shape
    assert output.last_hidden_state.shape == (1, 12, config.hidden_size), \
        f"Unexpected output shape: {output.last_hidden_state.shape}"

    print("Model forward pass successful with output shape:", output.last_hidden_state.shape)

if __name__ == "__main__":
    test_model_forward_pass()
