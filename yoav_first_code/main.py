import cv2
import torch
import numpy as np
from YOLOv8 import FaceDetector
from ResNet import build_model #LetterboxResize   Using your padded architecture
from low_light_agent import DynamicLowLightAgent
from motion_blur_agent import MotionBlurAgent
from low_res_agent import SuperResAgent
import json
from generate_data.generate_data_for_gate import letterbox_resize

# --- ⚙ CONFIGURATION ---
MODEL_PATHS = {
    "yolo": "models/yolov8n-face.pt",
    "resnet": "models/id_classifier_resnet18.pt",
    "mapping": "models/class_mapping.json",
    "sr_pb": "models/FSRCNN-small_x3.pb"
}


class IntegratedGate:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" Initializing System on: {self.device}")

        # 1. Initialize Face Detector (YOLOv8)
        self.detector = FaceDetector(model_path=MODEL_PATHS["yolo"])

        # 2. Initialize Identification Model (ResNet-18)
        with open(MODEL_PATHS["mapping"], 'r') as f:
            self.classes = [k for k, v in sorted(json.load(f).items(), key=lambda x: x[1])]

        self.id_model = build_model(num_classes=len(self.classes))
        self.id_model.load_state_dict(torch.load(MODEL_PATHS["resnet"], map_location=self.device))
        self.id_model.to(self.device).eval()

        # 3. Initialize Preprocessing Agents
        self.low_light_agent = DynamicLowLightAgent()
        self.motion_blur_agent = MotionBlurAgent(amount=1.2)
        self.super_res_agent = SuperResAgent(model_name=MODEL_PATHS["sr_pb"])

        # Standards
        self.letterbox = letterbox_resize(target_size=224)
        self.resnet_tfms = torch.nn.Sequential(
            # Standard ResNet-18 Normalization
        )

    def classify_quality(self, face_crop):
        """Logic to decide which agent is needed."""
        avg_brightness = np.mean(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY))
        h, w = face_crop.shape[:2]

        if avg_brightness < 60: return "low_light"
        if h < 80 or w < 80: return "low_res"

        # Optional: Add a Laplacian check for blur
        if cv2.Laplacian(face_crop, cv2.CV_64F).var() < 100: return "motion_blur"

        return "normal"

    def run(self):
        cap = cv2.VideoCapture(0)  # Camera Stream

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # STEP 1: Detect Faces (YOLO)
            results = self.detector.detect(frame, expand_ratio=0.20)

            for res in results:
                face = res["crop"]
                coords = res["coords"]  # [x1, y1, x2, y2]

                # STEP 2: Determine & Apply Correction
                quality = self.classify_quality(face)

                if quality == "low_light":
                    fixed_face = self.low_light_agent.process(face)
                elif quality == "low_res":
                    fixed_face = self.super_res_agent.process(face)
                elif quality == "motion_blur":
                    fixed_face = self.motion_blur_agent.process(face)
                else:
                    fixed_face = face

                # STEP 3: Identify Person (ResNet)
                # Apply padding to match your new training
                input_face = self.letterbox(fixed_face)

                # Convert to Tensor (Simplified for the example)
                tensor = torch.from_numpy(np.array(input_face)).permute(2, 0, 1).float().unsqueeze(0).to(
                    self.device) / 255.0

                with torch.no_grad():
                    output = self.id_model(tensor)
                    prob = torch.nn.functional.softmax(output, dim=1)
                    conf, pred = torch.max(prob, 1)

                # STEP 4: Draw Results
                label = f"{self.classes[pred.item()]} ({conf.item() * 100:.1f}%)"
                color = (0, 255, 0) if self.classes[pred.item()] != "other" else (0, 0, 255)

                cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)
                cv2.putText(frame, f"{label} | Mode: {quality}", (coords[0], coords[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Jetson Integrated Gate System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    gate = IntegratedGate()
    gate.run()
