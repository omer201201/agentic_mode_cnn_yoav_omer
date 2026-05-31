import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from tqdm import tqdm
# ----------------------------------------
# 1. Super Resolution Agent Definition
# PURPOSE: This agent receives low-resolution face crops and uses a lightweight
# Deep Neural Network (ESPCN) to intelligently upscale the image, hallucinating
# missing details rather than just stretching the pixels.
# ----------------------------------------
# --- CONFIGURATION ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = {
    "Espcn": os.path.join(PROJECT_ROOT, "models", "ESPCN_x3.pb")
}

class SuperResAgent:
    def __init__(self, model_path=MODEL_PATH["Espcn"], scale=3, algo_name="espcn"):

        self.scale = scale
        self.model_path = model_path

        # 1. Verify Model Exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[SuperRes] CRITICAL ERROR: Model file not found at '{self.model_path}'."
            )

        # 2. Load OpenCV Super Resolution
        try:
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()

            # Read the model file
            self.sr.readModel(self.model_path)

            # Set the specific algorithm and scale
            self.sr.setModel(algo_name, scale)

            print(f"[SuperRes] Success! Loaded {algo_name.upper()} model (Scale x{scale})")

        except Exception as e:
            raise RuntimeError(f"[SuperRes] Failed to load model. Error: {e}")

    def process(self, face_crop):

        if face_crop is None or face_crop.size == 0:
            return face_crop

        # 1. Run AI Inference
        # This will now use the sharper model ESPCN
        result = self.sr.upsample(face_crop)

        # Create a blurred version of the upscaled image
        gaussian = cv2.GaussianBlur(result, (0, 0), 2.0)

        # Blend the original upscaled image with the blurred version.
        # Formula: (original * 1.5) + (blurred * -0.5) + 0 ==Original + 0.5 * (Original - Blurred)
        # By subtracting a fraction of the blurred image, we isolate and boost the edge details.
        result = cv2.addWeighted(result, 1.5, gaussian, -0.5, 0)

        return result

    def process_directory(self, input_dir, output_dir):
        if not os.path.exists(input_dir):
            print(f"Error: Input directory '{input_dir}' does not exist.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')
        image_files = [f for f in os.listdir(input_dir) if f.endswith(valid_extensions)]

        if not image_files:
            print(f"No images found in {input_dir}.")
            return

        print(f"Found {len(image_files)} images. Starting Super Resolution batch process...")

        for filename in tqdm(image_files, desc="Upscaling Images"):
            img_path = os.path.join(input_dir, filename)

            img = cv2.imread(img_path)
            if img is None:
                continue

            try:
                # Apply Super Resolution
                upscaled_img = self.process(img)

                # Save the enhanced image
                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, upscaled_img)
            except Exception as e:
                print(f"\nError processing {filename}: {e}")

        print(f"\nFinished processing! All images saved to: {output_dir}")


def main():
    print("Testing Super Resolution Agent")
    # 1. Initialize the Agent
    try:
        agent = SuperResAgent(model_path=MODEL_PATH["Espcn"], scale=3, algo_name="espcn")
    except Exception as e:
        print(e)
        return

    print("\n--- Running Batch Process ---")
    input_folder = r"C:\Users\Your0124\pycharm_project_test\agentic_mode_cnn_yoav_omer-Organized_Project_12-04-2026\agentic_mode_cnn_yoav_omer-Organized_Project_12-04-2026\Organized_project_13_4\generate_data\other\train\low_res"
    output_folder = r"C:\Users\Your0124\pycharm_project_test\agentic_mode_cnn_yoav_omer-Organized_Project_12-04-2026\agentic_mode_cnn_yoav_omer-Organized_Project_12-04-2026\Organized_project_13_4\generate_data\other\train\low_res"

    agent.process_directory(input_folder, output_folder)

if __name__ == "__main__":
    main()
