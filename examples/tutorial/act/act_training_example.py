"""This script demonstrates how to train ACT Policy on a real-world dataset."""

from pathlib import Path

import torch

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features


def main():
    output_directory = Path("outputs/robot_learning_tutorial/act")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select your device
    device = torch.device("mps")  # or "cuda" or "cpu"

    dataset_id = "lerobot/svla_so101_pickplace"

    # This specifies the inputs the model will be expecting and the outputs it will produce
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    policy = ACTPolicy(cfg)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    policy.train()
    policy.to(device)

    # The policy declares what a training sample holds: ACT predicts `chunk_size` future actions.
    dataset = LeRobotDataset(dataset_id, sampler=cfg.training_sampler)

    # Create the optimizer and dataloader for offline training
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    batch_size = 32
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Number of training steps and logging frequency
    training_steps = 1
    log_freq = 1

    # Run training loop
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

    # Save the policy checkpoint, alongside the pre/post processors
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)

    # Save all assets to the Hub
    policy.push_to_hub("<user>/robot_learning_tutorial_act")
    preprocessor.push_to_hub("<user>/robot_learning_tutorial_act")
    postprocessor.push_to_hub("<user>/robot_learning_tutorial_act")


if __name__ == "__main__":
    main()
