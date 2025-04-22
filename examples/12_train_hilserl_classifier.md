# Training a HIL-SERL Reward Classifier with LeRobot

This tutorial provides step-by-step instructions for training a reward classifier using LeRobot.

---

## Training Script Overview

LeRobot includes a ready-to-use training script located at [`lerobot/scripts/train_hilserl_classifier.py`](../../lerobot/scripts/train_hilserl_classifier.py). Here's an outline of its workflow:

1. **Configuration Loading**
   The script uses Hydra to load a configuration file for subsequent steps. (Details on Hydra follow below.)

2. **Dataset Initialization**
   It loads a `LeRobotDataset` containing images and rewards. To optimize performance, a weighted random sampler is used to balance class sampling.

3. **Classifier Initialization**
   A lightweight classification head is built on top of a frozen, pretrained image encoder from HuggingFace. The classifier outputs either:
   - A single probability (binary classification), or
   - Logits (multi-class classification).

4. **Training Loop Execution**
   The script performs:
   - Forward and backward passes,
   - Optimization steps,
   - Periodic logging, evaluation, and checkpoint saving.

---

## Configuring with Hydra

For detailed information about Hydra usage, refer to [`examples/4_train_policy_with_script.md`](../examples/4_train_policy_with_script.md). However, note that training the reward classifier differs slightly and requires a separate configuration file.

### Config File Setup

The default `default.yaml` cannot launch the reward classifier training directly. Instead, you need a configuration file like [`lerobot/configs/policy/hilserl_classifier.yaml`](../../lerobot/configs/policy/hilserl_classifier.yaml), with the following adjustment:

Replace the `dataset_repo_id` field with the identifier for your dataset, which contains images and sparse rewards:

```yaml
# Example: lerobot/configs/policy/reward_classifier.yaml
dataset_repo_id: "my_dataset_repo_id"
## Typical logs and metrics
```
When you start the training process, you will first see your full configuration being printed in the terminal. You can check it to make sure that you config it correctly and your config is not overridden by other files. The final configuration will also be saved with the checkpoint.

After that, you will see training log like this one:

```
[2024-11-29 18:26:36,999][root][INFO] -
Epoch 5/5
Training:  82%|██████████████████████████████████████████████████████████████████████████████▋                 | 91/111 [00:50<00:09,  2.04it/s, loss=0.2999, acc=69.99%]
```

or evaluation log like:

```
Validation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 28/28 [00:20<00:00,  1.37it/s]
```

### Metrics Tracking with Weights & Biases (WandB)

If `wandb.enable` is set to `true`, the training and evaluation logs will also be saved in WandB. This allows you to track key metrics in real-time, including:

- **Training Metrics**:
  - `train/accuracy`
  - `train/loss`
  - `train/dataloading_s`
- **Evaluation Metrics**:
  - `eval/accuracy`
  - `eval/loss`
  - `eval/eval_s`

#### Additional Features

You can also log sample predictions during evaluation. Each logged sample will include:

- The **input image**.
- The **predicted label**.
- The **true label**.
- The **classifier's "confidence" (logits/probability)**.

These logs can be useful for diagnosing and debugging performance issues.


#### Generate protobuf files

```bash
python -m grpc_tools.protoc \
    -I lerobot/scripts/server \
    --python_out=lerobot/scripts/server \
    --grpc_python_out=lerobot/scripts/server \
    lerobot/scripts/server/hilserl.proto
```
