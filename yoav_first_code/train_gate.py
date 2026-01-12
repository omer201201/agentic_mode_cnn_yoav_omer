import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Subset

# Import your model
from gate import SimpleGateCNN


# ----------------------------------------
# 1. Data
# ----------------------------------------
def get_gate_dataloaders(data_dir: str, batch_size: int = 16, num_workers: int = 2):
    # 1. Define Transforms
    train_tfms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 2. Load the dataset TWICE (The "Two Album" Trick)
    # We load it first just to get the labels/targets
    full_train_style = datasets.ImageFolder(root=data_dir, transform=train_tfms)
    full_val_style = datasets.ImageFolder(root=data_dir, transform=val_tfms)

    # 3. Get the labels for every image
    # This gives us a list like [0, 0, 1, 1, 2, 3...] corresponding to classes
    targets = full_train_style.targets

    # 4. Stratified Split
    # We split the INDICES (0 to 599), not the images themselves yet.
    # stratify=targets ensures the mix of classes in train_idx and val_idx is identical.
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        shuffle=True,
        stratify=targets  # <--- This is the magic part
    )

    # 5. Map indices to the correct dataset version
    # Train indices -> Pull from the "Augmented/Scribbled" Album
    train_ds = Subset(full_train_style, train_idx)

    # Val indices -> Pull from the "Clean" Album
    valid_ds = Subset(full_val_style, val_idx)

    print(f"Total: {len(targets)}")
    print(f"Train: {len(train_ds)} (guaranteed balanced)")
    print(f"Valid: {len(valid_ds)} (guaranteed balanced)")

    # 6. Create Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, full_train_style.class_to_idx


# ----------------------------------------
# 2. Train / Eval Functions
# ----------------------------------------
def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        # Calculate accuracy
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ----------------------------------------
# 3. Main Training Manager
# ----------------------------------------
def train_gate_manager(model, train_loader, valid_loader, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.5,patience=2,threshold=0.01)
    best_acc = 0.0
    best_state = None

    print("\nStarting Training...")

    for epoch in range(epochs):
        # 1. Train
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)

        # 2. Evaluate
        va_loss, va_acc = evaluate(model, valid_loader, device, criterion)
        scheduler.step(va_acc)
        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train: loss={tr_loss:.4f} acc={tr_acc:.2%} | "
              f"Valid: loss={va_loss:.4f} acc={va_acc:.2%}")

        # 3. Save Best
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  ->  New Best Accuracy: {best_acc:.2%}")

    return model


# ----------------------------------------
# 4. Main Execution
# ----------------------------------------
def main():
    # Configuration
    data_dir = "data/gate_dataset"
    output_dir = "models"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1. Get Data
    train_loader, valid_loader, class_to_idx = get_gate_dataloaders(data_dir, batch_size=16)
    print("Classes:", class_to_idx)

    # 2. Build Model
    model = SimpleGateCNN().to(device)

    # 3. Train
    model = train_gate_manager(
        model,
        train_loader,
        valid_loader,
        device,
        epochs=15
    )

    # 4. Save Results
    model_path = os.path.join(output_dir, "gate_model_best.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved best model to: {model_path}")


if __name__ == "__main__":
    main()
