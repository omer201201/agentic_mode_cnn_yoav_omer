import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# ==========================================
# 1. The Neural Network
# ==========================================
class SimpleGateCNN(nn.Module):
    """
    Lightweight CNN with Batch Normalization.
    Structure: [Conv -> BN -> ReLU -> Pool] x 3 -> Flatten -> FC -> Dropout -> FC
    built as a hierarchical feature extractor
    """

    def __init__(self, dropout_prob=0.1):
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

        # Pooling layer shrink the image size while keeping important features
        self.pool = nn.MaxPool2d(2, 2)

        # --- Classification Head ---
        # 128 channels * 8 height * 8 width
        # FC Neurons increased 128 -> 256
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        # Block 1: Conv -> BN -> ReLU -> Pool
        x = self.conv1(x)   # find patterns
        x = self.bn1(x)     # to keep training stable
        x = F.relu(x)       # non-linear activation function
        x = self.pool(x)    # reduces the dimensions of the image by half each time

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

        # Transform for the CNN (The Gate sees 128X128 regardless of input size)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


    def resize_with_padding(self, img, target_size=128):
        """
        Resize image to target_size x target_size while keeping aspect ratio.
        Fills the empty space with black.
        """
        h, w = img.shape[:2]
        scale = min(target_size / h, target_size / w)

        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create black canvas
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

        # Center image
        x_off = (target_size - new_w) // 2
        y_off = (target_size - new_h) // 2

        #You "paste" the resized face into that specific black area.
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    def process(self, face_crop):
        """
        Input: A cropped FACE image (numpy array) from YOLO.
        Output: The class string.
        """
        if face_crop is None or face_crop.size == 0:
            return "error"

        # Hard Logic for tiny faces (Speed optimization)
        h, w = face_crop.shape[:2]
        if h < 30 or w < 30:
            return "low_res"

        # STEP 1: Apply Smart Resizing
        # We resize to 128x128 because that's what the model expects
        processed_face = self.resize_with_padding(face_crop, target_size=128)

        #  STEP 2: Convert to PyTorch format
        img_rgb = cv2.cvtColor(processed_face, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # --- STEP 3: Ask the Brain ---
        with torch.no_grad():
            outputs = self.model(input_tensor)

            # 1. Apply Softmax to turn raw numbers into probabilities (0.0 to 1.0)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # 2. Find the highest probability and its class index
            conf, predicted_idx = torch.max(probabilities, 1)

            predicted_class = self.classes[predicted_idx.item()]

            # 3. Now the percentage will always be between 0 and 100
            confidence_score = conf.item() * 100

        return  confidence_score,predicted_class

