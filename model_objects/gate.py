import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms
import numpy as np

# ------------------------------------------
#             1. The Neural Network
# ------------------------------------------
class SimpleGateCNN(nn.Module):

    def __init__(self, dropout_prob=0.2): # Slightly higher dropout for deeper network
        super(SimpleGateCNN, self).__init__()

        #          --- Convolutional Blocks ---
        # Each block extracts increasingly complex patterns.
        # BatchNorm normalizes the math, speeding up training and preventing dead neurons.


        # --- Block 1 ---
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm2d(32)

        # --- Block 2 ---
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # --- Block 3 ---
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        # --- Block 4 ---
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        # --- Global Average Pooling ---
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # --- Classification Head ---
        # The input is now 256 from the 4th block
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x) # 224 -> 112

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x) # 112 -> 56

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x) # 56 -> 28

        # Block 4 (New)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x) # 28 -> 14

        # GAP -> Flatten
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
# ------------------------------------------
# 2. The Gate Controller (Adjusted for YOLO)
# ------------------------------------------
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


    def smart_resize(self, img, target_size=224):
        h, w = img.shape[:2]

        # Determine if we are upscaling or downscaling
        if h < target_size or w < target_size:
            # We are enlarging the image
            # INTER_CUBIC provides the best smoothness for faces
            interp = cv2.INTER_CUBIC
        else:
            # We are shrinking the image
            # INTER_AREA is best for maintaining detail when downscaling
            interp = cv2.INTER_AREA

        return cv2.resize(img, (target_size, target_size), interpolation=interp)


    def process(self, face_crop):

        #Input: A cropped FACE image (numpy array) from YOLO.
        #Output: The class string.

        if face_crop is None or face_crop.size == 0:
            return "error"

        # Hard Logic for tiny faces low res check
        h, w = face_crop.shape[:2]
        if h < 40 or w < 40:
            return 100, "low_res"

        # 2. LOW LIGHT CHECK (Brightness)
        # Convert to LAB to isolate perceptual lightness from color.
        # This matches the logic you already use in your LowLightAgent.
        lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)
        avg_brightness = np.mean(l_channel)

        if avg_brightness < 45:  # Tune this based on your dataset
            return 100 , "low_light"
        
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # We apply a very slight Gaussian blur before the Laplacian check.
        # This smooths out minor sensor noise without destroying real structural edges.
        smoothed_gray = cv2.GaussianBlur(gray, (3, 3), 0)
        laplacian_var = cv2.Laplacian(smoothed_gray, cv2.CV_64F).var()
        '''
        if laplacian_var < 30:  # Lower variance = fewer sharp edges = blurry
            return 100 , "motion_blur"
        '''
        # --- DEEP LEARNING PATHWAY ---
        # STEP 1: Apply Smart Resizing
        processed_face = self.smart_resize(face_crop, target_size=224)

        #  STEP 2: Convert to PyTorch format
        img_rgb = cv2.cvtColor(processed_face, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

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

