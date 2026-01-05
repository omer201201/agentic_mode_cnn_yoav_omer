import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from gate import SimpleGateCNN  # Import the model structure you wrote


def train_gate_model():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 2. Data Preparation
    # We resize everything to 64x64 because the Gate needs to be FAST.
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # LOAD DATA FROM FOLDER
    # Make sure your path matches where you created the folders!
    dataset_path = "data/gate_dataset"

    try:
        train_data = datasets.ImageFolder(root=dataset_path, transform=transform)
        train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
        print(f"Found {len(train_data)} images across classes: {train_data.classes}")
    except Exception as e:
        print("Error: Could not find data. Did you create the 'gate_dataset' folders?")
        return

    # 3. Initialize Model
    model = SimpleGateCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop (We run 10 times over the data)
    epochs = 10
    print("Starting Training...")

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(train_loader):.4f}")

    # 5. Save the Brain
    save_path = "models/gate_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training Complete! Model saved to {save_path}")


if __name__ == "__main__":
    train_gate_model()