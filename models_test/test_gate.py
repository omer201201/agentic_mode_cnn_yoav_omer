import os
import cv2
from model_objects.gate import AdaptiveGate

# ----------------------------------------
# Gate Testing Pipeline
# PURPOSE: This script acts as a standalone diagnostic tool to test the
# AdaptiveGate model on a folder of raw images. It verifies that the gate
# correctly identifies the specific degradation (blur, low-res, dark, normal)
# before these images are sent to the repair agents.
# ----------------------------------------
# --- CONFIGURATION ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = {
    "GATE_WEIGHTS" : os.path.join(PROJECT_ROOT, "models", "gate_model_best_7.pth")
}

def run_gate_pipeline():
    # 1. SETUP
    folder_path = r"C:\Users\yoavt\PycharmProjects\final_projact\data\resnet dataset\real_test\test_for_gate"
    gate = AdaptiveGate(model_path=MODEL_PATH["GATE_WEIGHTS"])

    # 2. TRACKING
    # Dynamically create a dictionary to keep score based on whatever classes
    # the gate model was trained on (e.g., {'low_light': 0, 'normal': 0, ...})
    summary_counts = {cls: 0 for cls in gate.classes}

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg'))]
    total = len(image_files)

    print(f"-- Processing {total} images through the Gate ---")
    print(f"{'IMAGE NAME':<25} | {'DECISION':<12} | {'CONFIDENCE'}")
    print("-" * 60)

    for filename in image_files:
        # Load the raw image using OpenCV (loads as BGR)
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is None: continue
        # This call handles ALL the resizing and padding to 224X224
        confidence, decision = gate.process(img)

        # Update our summary counter
        summary_counts[decision] += 1
        print(f"{filename:<25} | {decision:<12} | {confidence:.1f}%")

    # 4. FINAL CLASS SUMMARY
    print("\n" + "=" * 35)
    print("      GATE RESULTS SUMMARY")
    print("=" * 35)
    for cls_name, count in summary_counts.items():
        print(f"{cls_name.upper():<15}: {count} images")
    print("-" * 35)
    print(f"TOTAL PROCESSED: {total}")
    print("=" * 35)

if __name__ == "__main__":
    run_gate_pipeline()