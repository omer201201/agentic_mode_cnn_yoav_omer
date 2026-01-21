import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import YOLOv8


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
        self.classes = ['low_light', 'low_res', 'motion_blur', 'normal']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the Brain
        self.model = SimpleGateCNN().to(self.device)
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print(f" Gate Model Loaded: {model_path}")
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
            return "error problem with yolo"

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
def main():
    print("--- 🚀 Starting Full Pipeline Test ---")

    # 1. Setup Paths
    # Adjust this path to point to your 'yoav' folder
    folder_path = os.path.join( "data", "train", "yoav")

    # 2. Initialize Agents (Load them ONCE)
    print("\n[1/3] Loading Agents...")
    try:
        # Load YOLO
        detector = YOLOv8.FaceDetector(model_path="models/yolov8n-face.pt")

        # Load Gate (Ensure you fixed the class list order in gate.py!)
        gate = AdaptiveGate(model_path="models/gate_model_best.pth")
        print(" Agents Loaded.")
    except Exception as e:
        print(f" Error loading models: {e}")
        return

    # 3. Get Images
    if not os.path.exists(folder_path):
        print(f" Folder not found: {folder_path}")
        return

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"[2/3] Found {len(image_files)} images to test.\n")

    # 4. Processing Loop
    results = {'normal': 0, 'low_light': 0, 'low_res': 0, 'motion_blur': 0}

    print(f"{'FILENAME':<25} | {'FACE SIZE':<15} | {'GATE DECISION'}")
    print("-" * 60)

    for filename in image_files:
        full_path = os.path.join(folder_path, filename)
        img = cv2.imread(full_path)

        if img is None: continue

        # --- STEP A: YOLO DETECTION ---
        faces = detector.detect(img)

        if len(faces) == 0:
            print(f"{filename:<25} | {'No Face':<15} | [SKIPPED]")
            continue

        # Get the first face found
        face_crop = faces[0]["crop"]
        h, w = face_crop.shape[:2]

        # --- STEP B: GATE CLASSIFICATION ---
        # The Gate now sees ONLY the face, no background noise
        decision = gate.process(face_crop)

        # Update Stats
        results[decision] += 1

        print(f"{filename:<25} | {f'{w}x{h}':<15} | {decision}")

    # 5. Final Report
    print("\n" + "=" * 30)
    print(" FINAL RESULTS")
    print("=" * 30)
    for category, count in results.items():
        print(f"{category.title()}: {count}")
    print("=" * 30)


if __name__ == "__main__":
    main()
