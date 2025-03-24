#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from lerobot.common.policies.hilserl.classifier.modeling_classifier import (
    Classifier,
    ClassifierConfig,
)

BATCH_SIZE = 1000
LR = 0.1
EPOCH_NUM = 2

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def train_evaluate_multiclass_classifier():
    logging.info(
        f"Start multiclass classifier train eval with {DEVICE} device, batch size {BATCH_SIZE}, learning rate {LR}"
    )
    multiclass_config = ClassifierConfig(model_name="microsoft/resnet-18", device=DEVICE, num_classes=10)
    multiclass_classifier = Classifier(multiclass_config)

    trainset = CIFAR10(root="data", train=True, download=True, transform=ToTensor())
    testset = CIFAR10(root="data", train=False, download=True, transform=ToTensor())

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    multiclass_num_classes = 10
    epoch = 1

    criterion = CrossEntropyLoss()
    optimizer = Adam(multiclass_classifier.parameters(), lr=LR)

    multiclass_classifier.train()

    logging.info("Start multiclass classifier training")

    # Training loop
    while epoch < EPOCH_NUM:  # loop over the dataset multiple times
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = multiclass_classifier(inputs)

            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:  # print every 10 mini-batches
                logging.info(f"[Epoch {epoch}, Batch {i}] loss: {loss.item():.3f}")

        epoch += 1

    print("Multiclass classifier training finished")

    multiclass_classifier.eval()

    test_loss = 0.0
    test_labels = []
    test_pridections = []
    test_probs = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = multiclass_classifier(images)
            loss = criterion(outputs.logits, labels)
            test_loss += loss.item() * BATCH_SIZE

            _, predicted = torch.max(outputs.logits, 1)
            test_labels.extend(labels.cpu())
            test_pridections.extend(predicted.cpu())
            test_probs.extend(outputs.probabilities.cpu())

    test_loss = test_loss / len(testset)

    logging.info(f"Multiclass classifier test loss {test_loss:.3f}")

    test_labels = torch.stack(test_labels)
    test_predictions = torch.stack(test_pridections)
    test_probs = torch.stack(test_probs)

    accuracy = Accuracy(task="multiclass", num_classes=multiclass_num_classes)
    precision = Precision(task="multiclass", average="weighted", num_classes=multiclass_num_classes)
    recall = Recall(task="multiclass", average="weighted", num_classes=multiclass_num_classes)
    f1 = F1Score(task="multiclass", average="weighted", num_classes=multiclass_num_classes)
    auroc = AUROC(task="multiclass", num_classes=multiclass_num_classes, average="weighted")

    # Calculate metrics
    acc = accuracy(test_predictions, test_labels)
    prec = precision(test_predictions, test_labels)
    rec = recall(test_predictions, test_labels)
    f1_score = f1(test_predictions, test_labels)
    auroc_score = auroc(test_probs, test_labels)

    logging.info(f"Accuracy: {acc:.2f}")
    logging.info(f"Precision: {prec:.2f}")
    logging.info(f"Recall: {rec:.2f}")
    logging.info(f"F1 Score: {f1_score:.2f}")
    logging.info(f"AUROC Score: {auroc_score:.2f}")


def train_evaluate_binary_classifier():
    logging.info(
        f"Start binary classifier train eval with {DEVICE} device, batch size {BATCH_SIZE}, learning rate {LR}"
    )

    target_binary_class = 3

    def one_vs_rest(dataset, target_class):
        new_targets = []
        for _, label in dataset:
            new_label = float(1.0) if label == target_class else float(0.0)
            new_targets.append(new_label)

        dataset.targets = new_targets  # Replace the original labels with the binary ones
        return dataset

    binary_train_dataset = CIFAR10(root="data", train=True, download=True, transform=ToTensor())
    binary_test_dataset = CIFAR10(root="data", train=False, download=True, transform=ToTensor())

    # Apply one-vs-rest labeling
    binary_train_dataset = one_vs_rest(binary_train_dataset, target_binary_class)
    binary_test_dataset = one_vs_rest(binary_test_dataset, target_binary_class)

    binary_trainloader = DataLoader(binary_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    binary_testloader = DataLoader(binary_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    binary_epoch = 1

    binary_config = ClassifierConfig(model_name="microsoft/resnet-50", device=DEVICE)
    binary_classifier = Classifier(binary_config)

    class_counts = np.bincount(binary_train_dataset.targets)
    n = len(binary_train_dataset)
    w0 = n / (2.0 * class_counts[0])
    w1 = n / (2.0 * class_counts[1])

    binary_criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(w1 / w0))
    binary_optimizer = Adam(binary_classifier.parameters(), lr=LR)

    binary_classifier.train()

    logging.info("Start binary classifier training")

    # Training loop
    while binary_epoch < EPOCH_NUM:  # loop over the dataset multiple times
        for i, data in enumerate(binary_trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(torch.float32).to(DEVICE)

            # Zero the parameter gradients
            binary_optimizer.zero_grad()

            # Forward pass
            outputs = binary_classifier(inputs)
            loss = binary_criterion(outputs.logits, labels)
            loss.backward()
            binary_optimizer.step()

            if i % 10 == 0:  # print every 10 mini-batches
                print(f"[Epoch {binary_epoch}, Batch {i}] loss: {loss.item():.3f}")
        binary_epoch += 1

    logging.info("Binary classifier training finished")
    logging.info("Start binary classifier evaluation")

    binary_classifier.eval()

    test_loss = 0.0
    test_labels = []
    test_pridections = []
    test_probs = []

    with torch.no_grad():
        for data in binary_testloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(torch.float32).to(DEVICE)
            outputs = binary_classifier(images)
            loss = binary_criterion(outputs.logits, labels)
            test_loss += loss.item() * BATCH_SIZE

            test_labels.extend(labels.cpu())
            test_pridections.extend(outputs.logits.cpu())
            test_probs.extend(outputs.probabilities.cpu())

    test_loss = test_loss / len(binary_test_dataset)

    logging.info(f"Binary classifier test loss {test_loss:.3f}")

    test_labels = torch.stack(test_labels)
    test_predictions = torch.stack(test_pridections)
    test_probs = torch.stack(test_probs)

    # Calculate metrics
    acc = Accuracy(task="binary")(test_predictions, test_labels)
    prec = Precision(task="binary", average="weighted")(test_predictions, test_labels)
    rec = Recall(task="binary", average="weighted")(test_predictions, test_labels)
    f1_score = F1Score(task="binary", average="weighted")(test_predictions, test_labels)
    auroc_score = AUROC(task="binary", average="weighted")(test_probs, test_labels)

    logging.info(f"Accuracy: {acc:.2f}")
    logging.info(f"Precision: {prec:.2f}")
    logging.info(f"Recall: {rec:.2f}")
    logging.info(f"F1 Score: {f1_score:.2f}")
    logging.info(f"AUROC Score: {auroc_score:.2f}")


if __name__ == "__main__":
    train_evaluate_multiclass_classifier()
    train_evaluate_binary_classifier()
