import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import torch
import numpy as np
import json
import time
import csv
import matplotlib.pyplot as plt
from model_objects.YOLOv8 import FaceDetector
from model_objects.ResNet import build_model
from model_objects.gate import AdaptiveGate
from restoration_agents.low_light_agent import DynamicLowLightAgent
from restoration_agents.motion_blur_agent import MotionBlurAgent
from restoration_agents.low_res_agent import SuperResAgent
from generate_data.generate_data_for_gate import smart_resize

# --- CONFIGURATION ---
TEST_FOLDER = r"C:\Users\yoavt\PycharmProjects\final_projact\data\system_test\yoav\low_light"
OUTPUT_CSV = "pipeline_comparison_low_light_yoav.csv"

MODEL_PATHS = {
    "yolo": r"C:\Users\yoavt\PycharmProjects\final_projact\models\yolov8n-face.pt",
    "resnet": r"C:\Users\yoavt\PycharmProjects\final_projact\models\resnet18_8.pt",
    "gate": r"C:\Users\yoavt\PycharmProjects\final_projact\models\gate_model_best_3.pth",
    "mapping": r"C:\Users\yoavt\PycharmProjects\final_projact\models\class_mapping.json"
}


class PipelineBenchmark:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Benchmark on: {self.device} ")

        # 1. Shared Models
        print("Loading Models...")
        self.detector = FaceDetector(model_path=MODEL_PATHS["yolo"])
        self.gate = AdaptiveGate(model_path=MODEL_PATHS["gate"])

        # Agents
        self.low_light_agent = DynamicLowLightAgent()
        self.motion_blur_agent = MotionBlurAgent()
        self.super_res_agent = SuperResAgent()
        self.smart_resize = smart_resize

        # ResNet Identification
        with open(MODEL_PATHS["mapping"], 'r') as f:
            self.classes = [k for k, v in sorted(json.load(f).items(), key=lambda x: x[1])]

        self.id_model = build_model(num_classes=len(self.classes))

        # Load weights safely
        try:
            self.id_model.load_state_dict(torch.load(MODEL_PATHS["resnet"], map_location=self.device))
        except:
            print("cant load resnet...")

        self.id_model.to(self.device).eval()

        # Normalization Stats
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def run_inference_resnet(self, face_img):
        # Preprocess
        input_face = self.smart_resize(face_img, target_size=224)
        input_face_rgb = cv2.cvtColor(input_face, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(np.array(input_face_rgb)).permute(2, 0, 1).float().unsqueeze(0).to(
            self.device) / 255.0
        tensor = (tensor - self.mean) / self.std

        # Predict
        with torch.no_grad():
            output = self.id_model(tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)

        # Unknown Threshold
        if conf.item() < 0.40:
            return "Unknown", conf.item()
        return self.classes[pred.item()], conf.item()

    def run_benchmark(self):
        headers = [
            "Image", "Face_Index","BASE_Name",
             "BASE_Conf", "BASE_Time(ms)","Gate Decision",
            "GATE_Name", "GATE_Conf", "GATE_Time(ms)",
            "Conf_Gain", "Time_Cost(ms)"
        ]

        image_files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.png', '.jpg'))]
        print(f"Testing {len(image_files)} Images ")

        # --- NEW: Data Tracking Variables ---
        gate_decisions = {"low_light": 0, "low_res": 0, "motion_blur": 0, "normal": 0}
        
        flips_good_to_bad = 0
        flips_bad_to_good = 0
        
        conf_changes_both_wrong = []
        conf_changes_both_right = []
        
        total_base_time = 0
        total_gate_time = 0
        total_faces_processed = 0
        # ------------------------------------
        # --- NEW: Success Trackers ---
        base_correct_count = 0
        gate_correct_count = 0
        # ------------------------------------

        with open(OUTPUT_CSV, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for filename in image_files:
                path = os.path.join(TEST_FOLDER, filename)
                frame = cv2.imread(path)
                if frame is None: continue

                # --- NEW: Extract Ground Truth from filename ---
                # Default to "other", override if "yoav" or "omer" is in the filename
                ground_truth = "other"
                if "yoav" in filename.lower():
                    ground_truth = "yoav"
                elif "omer" in filename.lower():
                    ground_truth = "omer"
                # -----------------------------------------------

                results = self.detector.detect(frame, expand_ratio=0.30)

                for res in results:
                    face = res["crop"]
                    coords = res["coords"]

                    # --- PATH A: BASELINE ---
                    t0 = time.time()
                    base_name, base_conf = self.run_inference_resnet(face)
                    base_time = (time.time() - t0) * 1000

                    # --- PATH B: INTEGRATED GATE ---
                    t2 = time.time()
                    gate_score, quality = self.gate.process(face)

                    # Track gate decision
                    if quality in gate_decisions:
                        gate_decisions[quality] += 1

                    if quality == "low_light":
                        fixed_face = self.low_light_agent.process(face)
                    elif quality == "low_res":
                        fixed_face = self.super_res_agent.process(face)
                    elif quality == "motion_blur":
                        fixed_face = self.motion_blur_agent.process(face)
                    else:
                        fixed_face = face

                    gate_name, gate_conf = self.run_inference_resnet(fixed_face)
                    gate_time = (time.time() - t2) * 1000

                    # --- NEW: Calculate Analytics ---
                    total_faces_processed += 1
                    total_base_time += base_time
                    total_gate_time += gate_time

                    base_is_correct = (base_name == ground_truth)
                    gate_is_correct = (gate_name == ground_truth)

                    # --- NEW: Track Success ---
                    if base_is_correct: base_correct_count += 1
                    if gate_is_correct: gate_correct_count += 1

                    # 2. Count flips
                    if base_is_correct and not gate_is_correct:
                        flips_good_to_bad += 1
                    elif not base_is_correct and gate_is_correct:
                        flips_bad_to_good += 1

                    # 3. Track Certainty Changes
                    conf_diff = gate_conf - base_conf
                    if not base_is_correct and not gate_is_correct:
                        conf_changes_both_wrong.append(conf_diff)
                    elif base_is_correct and gate_is_correct:
                        conf_changes_both_right.append(conf_diff)
                    # --------------------------------

                    print(f"[{filename[:10]} (Face {coords})] {quality.upper():<10} | Base: {base_conf:.2f} | Gate: {gate_conf:.2f}")

                    writer.writerow([
                        filename, coords,base_name,
                         f"{base_conf:.4f}", f"{base_time:.2f}",quality,
                        gate_name, f"{gate_conf:.4f}", f"{gate_time:.2f}",
                        f"{conf_diff*100:.2f}", f"{gate_time - base_time:.2f}"
                    ])

# ... [Keep your existing for-loop above this] ...

        print(f"\n Benchmark Complete! Results saved to: {os.path.abspath(OUTPUT_CSV)}")

        # --- Calculate Averages ---
        if total_faces_processed > 0:
            avg_base_time = total_base_time / total_faces_processed
            avg_gate_time = total_gate_time / total_faces_processed
            avg_right = (sum(conf_changes_both_right) / len(conf_changes_both_right) * 100) if conf_changes_both_right else 0
            avg_wrong = (sum(conf_changes_both_wrong) / len(conf_changes_both_wrong) * 100) if conf_changes_both_wrong else 0

            # --- NEW: Calculate Accuracy Percentages ---
            base_accuracy = (base_correct_count / total_faces_processed) * 100
            gate_accuracy = (gate_correct_count / total_faces_processed) * 100

            # --- NEW: Append Summary Table to the CSV ---
            with open(OUTPUT_CSV, mode='a', newline='') as f:
                writer = csv.writer(f)
                
                # Create a visual break (empty rows)
                writer.writerow([])
                writer.writerow([])
                

                # Write the new table headers and data
                writer.writerow(["--- SUMMARY STATISTICS ---", ""])
                writer.writerow(["Metric", "Value"])
                writer.writerow(["Total Faces Processed", total_faces_processed])
                
                # --- NEW: CSV Accuracy Rows ---
                writer.writerow(["Basic Pipeline Accuracy", f"{base_accuracy:.2f}%"])
                writer.writerow(["Gated Pipeline Accuracy", f"{gate_accuracy:.2f}%"])
                writer.writerow(["Net Accuracy Change", f"{gate_accuracy - base_accuracy:+.2f}%"])
                # ------------------------------
                writer.writerow(["Average Time Basic (ms)", f"{avg_base_time:.2f}"])
                writer.writerow(["Average Time Gated (ms)", f"{avg_gate_time:.2f}"])
                writer.writerow(["Time Cost of Gate (ms)", f"{avg_gate_time - avg_base_time:.2f}"])
                writer.writerow(["Flips: Bad to Good (Fixed)", flips_bad_to_good])
                writer.writerow(["Flips: Good to Bad (Ruined)", flips_good_to_bad])
                writer.writerow(["Avg Conf Change (Both Correct)", f"{avg_right:+.2f}%"])
                writer.writerow(["Avg Conf Change (Both Wrong)", f"{avg_wrong:+.2f}%"])

            print("Summary statistics appended to the CSV.")

        # --- 1. Create the Pie Chart ---
        # Filter out decisions with 0 counts to make the pie chart cleaner
        labels = [k for k, v in gate_decisions.items() if v > 0]
        sizes = [v for k, v in gate_decisions.items() if v > 0]
        
        if sizes:
            plt.figure(figsize=(8, 6))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
            plt.title('Gate Decisions Breakdown')
            plt.axis('equal') 
            plt.savefig('gate_decisions_pie_low_light_yoav.png')
            print("\nPie chart saved as 'gate_decisions_pie.png'. Closing the popup window will end the script.")
            plt.show() 

if __name__ == "__main__":
    benchmark = PipelineBenchmark()
    benchmark.run_benchmark()
