import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

import torch.nn as nn
import torchvision.models as models

import torchvision.transforms.v2 as v2

# =====================================================================
# =====================================================================

class StateClissifierDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # (C, H, W)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")

        image = Image.open(img_path).convert("RGB")

        label = [0]*4

        with open(label_path, 'r') as f:
            idx = int(f.read())
            label[idx] = 1

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return self.transform(image), label_tensor
    
# ------------------------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform2 = v2.Compose([
    v2.Resize((224,224)),
    v2.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


# ------------------------------------------------------------------------------

class StateClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4),  # 4 ориентации (возвращает логиты)
        )
        self.model = base

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        self.eval() # Переводим модель в режим оценки (отключает Dropout)
        with torch.no_grad(): # Отключаем расчет градиентов для скорости и памяти
            logits = self.forward(x)
            
            probabilities = torch.softmax(logits, dim=1)
            
            confidence, predicted_class = torch.max(probabilities, dim=1)
            
        return predicted_class, confidence
    
# =====================================================================
# =====================================================================

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, device='cuda', save_dir="./checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            targets = labels.argmax(dim=1)

            outputs = model(imgs)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(dim=1) == targets).sum().item()

        train_acc = train_correct / len(train_loader.dataset)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validation"):
                imgs, labels = imgs.to(device), labels.to(device)
                targets = labels.argmax(dim=1)
                outputs = model(imgs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(dim=1) == targets).sum().item()

        val_acc = val_correct / len(val_loader.dataset)

        print(f"\nEpoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}\n")

        last_model_path = os.path.join(save_dir, "last_model.pt")
        torch.save(model.state_dict(), last_model_path)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model at epoch {epoch+1} with val_acc = {val_acc:.4f}")
    
    c = 0
    for i,l in val_loader:
        print(l)
        i, l = i.to(device), l.to(device)
        o = model(i)
        print(o)
        c+=1
        if c > 10:
            break

