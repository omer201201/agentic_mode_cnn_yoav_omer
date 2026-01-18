import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


# ==========================================
# 1. The Neural Network (Same as before)
# ==========================================
class SimpleGateCNN(nn.Module):
    """
    Lightweight CNN with Batch Normalization.
    Structure: [Conv -> BN -> ReLU -> Pool] x 3 -> Flatten -> FC -> Dropout -> FC
    """

    def __init__(self, dropout_prob=0.5):
        super(SimpleGateCNN, self).__init__()

        # --- Block 1 (Doubled: 16 -> 32) ---
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # --- Block 2 (Doubled: 32 -> 64) ---
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # --- Block 3 (Doubled: 64 -> 128) ---
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        # Pooling layer (reused)
        self.pool = nn.MaxPool2d(2, 2)

        # --- Classification Head ---
        # 128 channels * 8 height * 8 width
        # FC Neurons increased 128 -> 256
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        # Block 1: Conv -> BN -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Block 2: Conv -> BN -> ReLU -> Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Block 3: Conv -> BN -> ReLU -> Pool
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten: Turn 3D volume into 1D vector
        x = x.view(x.size(0), -1)

        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
# ==========================================
# 2. The Gate Controller (Adjusted for YOLO)
# ==========================================
class AdaptiveGate:
    def __init__(self, model_path=None):
        self.classes = ['normal', 'low_light', 'motion_blur', 'low_res']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the Brain
        self.model = SimpleGateCNN().to(self.device)
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print(f"✅ Gate Model Loaded: {model_path}")
            except:
                print("️ Warning: Could not load weights. Using random weights.")

        # Transform for the CNN (The Gate sees 64x64 regardless of input size)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def process(self, face_crop):
        """
        Input: A cropped FACE image (numpy array) from YOLO.
        Output: The class string.
        """
        # --- Safety Check 1: Is the crop valid? ---
        if face_crop is None or face_crop.size == 0:
            return "error"

        # --- Safety Check 2: Is the face tiny? (Hard Logic) ---
        # Even if the CNN thinks it's "Normal", a 20x20 pixel face
        # is useless for ResNet (which wants 224x224).
        # We catch this BEFORE the CNN to save speed.
        h, w = face_crop.shape[:2]
        if h < 40 or w < 40:  # Threshold for "Too Small to even process"
            return "low_res"

        # --- Step A: Prepare for CNN ---
        # YOLO gives BGR (OpenCV format), PyTorch wants RGB
        img_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # --- Step B: Ask the Brain ---
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = self.classes[predicted_idx.item()]

        return predicted_class

    # ==========================================
    # 3. Simple Test Block
    # ==========================================


# ==========================================
# 3. Test Block (Run this file directly to test)
# ==========================================
if __name__ == "__main__":
    print("--- Testing Adaptive Gate (YOLO Mode) ---")

    # 1. Initialize Gate
    # Note: We removed 'target_size' because the Gate doesn't resize anymore
    gate = AdaptiveGate()

    # 2. Test with a "Normal" sized dummy face (100x100 black square)
    dummy_face = np.zeros((100, 100, 3), dtype=np.uint8)

    # We only get one variable back now (the decision)
    decision = gate.process(dummy_face)

    print(f"\nTest 1 (Normal Input):")
    print(f"Input Shape: {dummy_face.shape}")
    print(f"Gate Decision: {decision}")  # Will be random (e.g., 'normal' or 'blur') until trained

    # 3. Test with a "Tiny" face (30x30) to check the hard-coded safety check
    tiny_face = np.zeros((30, 30, 3), dtype=np.uint8)
    decision_tiny = gate.process(tiny_face)

    print(f"\nTest 2 (Tiny Input):")
    print(f"Input Shape: {tiny_face.shape}")
    print(f"Gate Decision: {decision_tiny}")  # Should ALWAYS be 'low_res'
