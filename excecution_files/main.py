import sys
import os
import cv2
import torch
import numpy as np
import json
from model_objects.YOLOv8 import FaceDetector
from model_objects.ResNet import build_model
from restoration_agents.low_light_agent import DynamicLowLightAgent
from restoration_agents.motion_blur_agent import MotionBlurAgent
from restoration_agents.low_res_agent import SuperResAgent
from generate_data.generate_data_for_gate import smart_resize
from model_objects.gate import AdaptiveGate
import time

# ----------------------------------------
# 1. Configuration & Global Paths
# ----------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

MODEL_PATHS = {
    "yolo": os.path.join(PROJECT_ROOT, "models", "yolov8n-face.pt"),
    "resnet": os.path.join(PROJECT_ROOT, "models", "resnet18_11.pt"),
    "gate": os.path.join(PROJECT_ROOT, "models", "gate_model_best_7.pth"),
    "mapping": os.path.join(PROJECT_ROOT, "models", "class_mapping.json"),
    "sr_pb": os.path.join(PROJECT_ROOT, "models", "ESPCN_x3.pb")
}

# ------------------------------------------
# The central nervous system of the project. It acts as the traffic
# controller, passing data from YOLO -> Gate -> Restoration Agent -> ResNet.
# ------------------------------------------

class IntegratedGate:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" Initializing Full Pipeline on: {self.device}")

        # 1. Face Detector (YOLOv8)
        self.detector = FaceDetector(model_path=MODEL_PATHS["yolo"])

        # 2. THE BRAIN: Adaptive Gate
        self.gate = AdaptiveGate(model_path=MODEL_PATHS["gate"])

        # 3. Identification Model (ResNet-18)
        with open(MODEL_PATHS["mapping"], 'r') as f:
            self.classes = [k for k, v in sorted(json.load(f).items(), key=lambda x: x[1])]

        # 4. Build the ResNet architecture and inject the custom weights
        self.id_model = build_model(num_classes=len(self.classes))
        self.id_model.load_state_dict(torch.load(MODEL_PATHS["resnet"], map_location=self.device))
        self.id_model.to(self.device).eval()

        # 5. Pre-calculated Normalization Tensors
        # Matches ResNet18_Weights
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        # 6. Preprocessing Agents
        self.low_light_agent = DynamicLowLightAgent()
        self.motion_blur_agent = MotionBlurAgent()
        self.super_res_agent = SuperResAgent()

        # Standardization for ResNet Identification (224x224)
        self.id_letterbox = smart_resize

    # ----------------------------------------------------------------------------
    # MODE 1: Live Camera Stream
    # PURPOSE: End-to-end real-time processing optimized for video feeds.
    # ----------------------------------------------------------------------------

    def run_live(self):
        cap = cv2.VideoCapture(0)  # Camera Stream
        print("--- System Live. Press 'q' to quit ---")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # STEP 1: Detect Faces (YOLO)
            results = self.detector.detect(frame, expand_ratio=0.40)

            for res in results:
                face = res["crop"]
                coords = res["coords"]

                # STEP 2: The Gate Decision
                gate_conf, quality = self.gate.process(face)

                # STEP 3: Route to Correct Agent
                if quality == "low_light":
                    fixed_face = self.low_light_agent.process(face)
                elif quality == "low_res":
                    fixed_face = self.super_res_agent.process(face)
                elif quality == "motion_blur":
                    fixed_face = self.motion_blur_agent.process(face)
                else:
                    fixed_face = face  # "normal" class passes through

                # STEP 4: Identify Person (ResNet)
                input_face = self.id_letterbox(fixed_face, target_size=224)

                # CRITICAL: BGR to RGB swap
                input_face_rgb = cv2.cvtColor(input_face, cv2.COLOR_BGR2RGB)

                # Tensor Conversion + Normalization
                tensor = torch.from_numpy(np.array(input_face_rgb)).permute(2, 0, 1).float().unsqueeze(0).to(
                    self.device) / 255.0
                tensor = (tensor - self.mean) / self.std

                # Run the final inference
                with torch.no_grad():
                    output = self.id_model(tensor)
                    prob = torch.nn.functional.softmax(output, dim=1)
                    id_conf, pred = torch.max(prob, 1)

                # Threshold Check for Unknowns
                if id_conf.item() < 0.60:
                    person_name = "Unknown"
                else:
                    person_name = self.classes[pred.item()]

                # STEP 5: Robust UI Drawing
                id_percent = id_conf.item() * 100
                label = f"{person_name} ({id_percent:.1f}%)"
                mode_text = f"Mode: {quality} ({gate_conf:.1f}%)"

                # Green for recognized, Red for 'other' or 'Unknown'
                color = (0, 255, 0) if person_name not in ["other", "Unknown"] else (0, 0, 255)

                # 1. Bounding Box
                cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)

                # 2. Text Positioning (Inside if too close to top)
                text_y_base = coords[1] + 25 if coords[1] < 60 else coords[1] - 15

                # 3. Background plate for readability
                cv2.rectangle(frame, (coords[0], text_y_base - 20), (coords[0] + 210, text_y_base + 25), (0, 0, 0), -1)

                # 4. Text Overlay
                cv2.putText(frame, label, (coords[0] + 5, text_y_base),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, mode_text, (coords[0] + 5, text_y_base + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Jetson Adaptive Gate Pipeline", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

    # ----------------------------------------------------------------------------
    # MODE 2: Folder Processing (Batch Diagnostics)
    # Loops through static images to measure pipeline accuracy and speed.
    # ----------------------------------------------------------------------------
    def run_on_folder(self, folder_path, output_folder="baseline_with_gate"):
        #Loops through every image in a folder and saves the visual results.

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            # CLEANUP BLOCK: Delete existing result images
            print(f"Clearing old results in '{output_folder}' ")
            for old_file in os.listdir(output_folder):
                old_file_path = os.path.join(output_folder, old_file)
                try:
                    if os.path.isfile(old_file_path):
                        os.remove(old_file_path)
                except Exception as e:
                    print(f"Could not delete {old_file}: {e}")

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Processing {len(image_files)} images from: {folder_path}")

        for filename in image_files:
            full_path = os.path.join(folder_path, filename)
            frame = cv2.imread(full_path)
            if frame is None: continue

            # --- Start Performance Timer ---
            t_start = time.time()

            # STEP 1: Detect Faces (YOLO)
            results = self.detector.detect(frame, expand_ratio=0.40)

            for res in results:

                face = res["crop"]
                coords = res["coords"]

                # STEP 2: The Gate Decision
                gate_conf, quality = self.gate.process(face)

                # STEP 3: Route to Correct Agent
                if quality == "low_light":
                    print("low_light")
                    fixed_face = self.low_light_agent.process(face)
                elif quality == "low_res":
                    print("low_res")
                    fixed_face = self.super_res_agent.process(face)
                elif quality == "motion_blur":
                    print("motion_blur")
                    fixed_face = self.motion_blur_agent.process(face)
                else:
                    fixed_face = face  # "normal" class
                    print("normal")

                # STEP 4: Identify Person (ResNet)
                input_face = self.id_letterbox(fixed_face, target_size=224)

                input_face_rgb = cv2.cvtColor(input_face, cv2.COLOR_BGR2RGB)

                tensor = torch.from_numpy(np.array(input_face_rgb)).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0

                # Apply ImageNet Normalization
                tensor = (tensor - self.mean) / self.std

                with torch.no_grad():
                    output = self.id_model(tensor)
                    prob = torch.nn.functional.softmax(output, dim=1)
                    id_conf, pred = torch.max(prob, 1)

                # Threshold Check
                if id_conf.item() < 0.60:
                    person_name = "Unknown"
                else:
                    person_name = self.classes[pred.item()]

                # --- End Performance Timer ---
                t_end = time.time()
                total_time = t_end - t_start
                print(f"️ Total Pipeline Time: {total_time:.4f} sec")

                # --- STEP 5: Robust Labeling ---
                id_percent = id_conf.item() * 100
                label = f"{person_name} ({id_percent:.1f}%)"
                mode_text = f"Mode: {quality} ({gate_conf:.1f}%)"

                color = (0, 255, 0) if person_name not in ["other", "Unknown"] else (0, 0, 255)

                cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)


                if coords[1] < 60:
                    text_y_base = coords[1] + 25
                else:
                    text_y_base = coords[1] - 1

                # 4. Draw the text labels
                cv2.putText(frame, label, (coords[0] + 5, text_y_base),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, mode_text, (coords[0] + 5, text_y_base + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


            # Save the result to new folder
            save_path = os.path.join(output_folder, f"result_{filename}")
            cv2.imwrite(save_path, frame)
            print(f"Saved: {save_path}")

        print(f"\n All results saved in the '{output_folder}' folder.")


if __name__ == "__main__":
    system = IntegratedGate()
    # OPTION 1: Run on images (Diagnostic Mode)
    # test_folder = os.path.join(PROJECT_ROOT, "data", "omer", "normal")
    # system.run_on_folder(test_folder

    # OPTION 2: Standard Video Stream (Use this later on Jetson Orin Nano)
    system.run_live()
