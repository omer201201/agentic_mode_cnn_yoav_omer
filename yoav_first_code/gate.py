import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


# ==========================================
# 1. The Neural Network Structure (The Brain)
# ==========================================
class SimpleGateCNN(nn.Module):
    """
    A lightweight CNN built from scratch.
    Designed for speed on Edge devices (Jetson).
    Input: 64x64 Color Image
    Output: 4 Classes (Normal, LowLight, Blur, LowRes)
    """

    def __init__(self):
        super(SimpleGateCNN, self).__init__()

        # Layer 1: Detect basic features (edges, colors)
        # Input channels=3 (RGB), Output=16, Kernel=3x3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Reduces size by half

        # Layer 2: Detect shapes
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Layer 3: Detect complex textures (blur/noise patterns)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Fully Connected Layer: Make the decision
        # After 3 pooling layers, a 64x64 image becomes 8x8.
        # So: 64 channels * 8 * 8 = 4096 inputs
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 Output Classes

    def forward(self, x):
        # Pass through layers with ReLU activation
        x = self.pool(F.relu(self.conv1(x)))  # 64 -> 32
        x = self.pool(F.relu(self.conv2(x)))  # 32 -> 16
        x = self.pool(F.relu(self.conv3(x)))  # 16 -> 8

        # Flatten for the decision layer
        x = x.view(-1, 64 * 8 * 8)

        # Classification
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ==========================================
# 2. The Gate Controller (The Manager)
# ==========================================
class AdaptiveGate:
    def __init__(self, model_path=None, target_size=(224, 224)):
        """
        model_path: Path to the trained .pth file (optional for now)
        target_size: The input size your Face Recognition model expects (112x112 or 224x224)
        """
        self.classes = ['normal', 'low_light', 'motion_blur', 'low_res']
        self.target_size = target_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the CNN
        self.model = SimpleGateCNN().to(self.device)

        # Load weights if we have them (We will train this later)
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()  # Set to inference mode
                print(f"✅ Gate Model Loaded: {model_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not load weights. using random weights. {e}")
        else:
            print("ℹ️ Note: Gate is running with RANDOM weights (needs training).")

        # Transformations for the Gate (Standardize input)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Gate uses small images for speed
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def process(self, cv2_image):
        """
        Main function to call from your pipeline.
        Input: Raw OpenCV image.
        Output: String ('normal', 'low_light', 'motion_blur', 'low_res')
        """
        if cv2_image is None:
            return "error"

        # --- Step A: Prepare for Classification ---
        # Convert BGR (OpenCV) to RGB (PyTorch)
        img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Prepare tensor for the Gate CNN
        # (The Gate uses 64x64 internally for speed, regardless of input size)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # --- Step B: Inference (Ask the Brain) ---
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = self.classes[predicted_idx.item()]

        return predicted_class


# ==========================================
# 3. Simple Test Block
# ==========================================
if __name__ == "__main__":
    # Create a dummy image (black square)
    dummy_img = np.zeros((500, 500, 3), dtype=np.uint8)

    # Initialize Gate
    gate = AdaptiveGate(target_size=(112, 112))

    # Run
    decision = gate.process(dummy_img)

    print(f"\n--- Gate Result ---")
    print(f"Decision: {decision}")  # Will be random until trained
