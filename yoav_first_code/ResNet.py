import os
import json
import numpy as np
import cv2
from pathlib import Path
from collections import Counter
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights


# ----------------------------------------
# 1. Padding Helper (Letterboxing)
# ----------------------------------------
class LetterboxResize:
    """
    Custom transform to resize image with padding (letterboxing)
    to maintain aspect ratio and prevent squashing.
    """

    def __init__(self, target_size=224):
        self.target_size = target_size

    def __call__(self, img):
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        h, w = img_cv.shape[:2]
        scale = min(self.target_size / h, self.target_size / w)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        x_offset = (self.target_size - new_w) // 2
        y_offset = (self.target_size - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        # Convert back to PIL for remaining transforms
        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


# ----------------------------------------
# 2. Data Loading & Transforms
# ----------------------------------------
def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 2):
    # Same mean/std as pre-trained weights
    mean = ResNet18_Weights.IMAGENET1K_V1.transforms().mean
    std = ResNet18_Weights.IMAGENET1K_V1.transforms().std

    # UPDATED: Added LetterboxResize to prevent squashing
    train_tfms = transforms.Compose([
        LetterboxResize(target_size=224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.05),  # Increased for robustness
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    valid_tfms = transforms.Compose([
        LetterboxResize(target_size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_ds = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=train_tfms)
    valid_ds = datasets.ImageFolder(root=os.path.join(data_dir, "valid"), transform=valid_tfms)

    if train_ds.class_to_idx != valid_ds.class_to_idx:
        raise ValueError("Class mappings do not match between train and valid folders.")

    # Sampler to handle dataset imbalance (e.g., more 'Other' than 'Yoav')
    train_counts = Counter(train_ds.targets)
    class_counts = torch.tensor([train_counts[i] for i in range(len(train_ds.classes))], dtype=torch.float)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[torch.tensor(train_ds.targets, dtype=torch.long)]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                              pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, train_ds.class_to_idx


# ----------------------------------------
# 3. Model & Training Components
# ----------------------------------------
def build_model(num_classes: int):
    model = models.resnet18(weights=None)
    # Load your local pre-trained weights
    state_dict = torch.load("models/resnet18-f37072fd.pth", map_location="cpu")
    model.load_state_dict(state_dict)

    # Modify the final Fully Connected layer for Yoav/Omer/Other
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_trainable_params(model: nn.Module, phase: int):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    if phase >= 2:
        # Fine-tune the deeper features for face recognition
        for p in model.layer4.parameters():
            p.requires_grad = True


def get_optimizer(model: nn.Module, phase: int):
    lr = 1e-3 if phase == 1 else 3e-5
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(params, lr=lr, weight_decay=1e-4)


# -------------------------------------
# 4. Training Engine
# -------------------------------------
def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def train_model(model, train_loader, valid_loader, device, p1_epochs=20, p2_epochs=10):
    criterion = nn.CrossEntropyLoss()
    best_acc, best_state = 0.0, None

    for phase in [1, 2]:
        print(f"\n--- Starting Phase {phase} ---")
        set_trainable_params(model, phase=phase)
        optimizer = get_optimizer(model, phase=phase)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        epochs = p1_epochs if phase == 1 else p2_epochs

        for epoch in range(epochs):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
            va_loss, va_acc = evaluate(model, valid_loader, device, criterion)
            scheduler.step(va_loss)

            print(f"Epoch {epoch + 1}/{epochs} | Train Acc: {tr_acc:.2%} | Valid Acc: {va_acc:.2%}")

            if va_acc > best_acc:
                best_acc = va_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)
    return model


# ------------------------------------
# 5. Execution Logic
# ------------------------------------
def main():
    data_dir = "data/resnet_dataset"
    output_dir = "models"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader, valid_loader, class_to_idx = get_dataloaders(data_dir)
    model = build_model(num_classes=len(class_to_idx)).to(device)

    model = train_model(model, train_loader, valid_loader, device)

    # Save final artifacts
    torch.save(model.state_dict(), os.path.join(output_dir, "id_classifier_resnet18.pt"))
    with open(os.path.join(output_dir, "class_mapping.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print("Training complete. Artifacts saved in /models")


if __name__ == "__main__":
    main()
