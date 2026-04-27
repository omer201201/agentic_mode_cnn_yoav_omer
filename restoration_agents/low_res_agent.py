import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

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
        # Formula: (original * 1.5) + (blurred * -0.5) + 0
        # By subtracting a fraction of the blurred image, we isolate and boost the edge details.
        result = cv2.addWeighted(result, 1.5, gaussian, -0.5, 0)

        return result



def main():
    print(" Testing Better Super Resolution")

    try:
        agent = SuperResAgent(model_path=MODEL_PATH["Espcn"], scale=3, algo_name="espcn")
    except Exception as e:
        print(e)
        return

    # Load  low-res image
    img_path = r"C:\Users\Your0124\final_project\Organized_project\data\yoav\low_res\yoav_low_res_55.jpg"
    if not os.path.exists(img_path):
        print("Image not found")
        return

    original = cv2.imread(img_path)
    upscaled = agent.process(original)

    # Visualization Code
    h, w = upscaled.shape[:2]

    original_resized = cv2.resize(original, (w, h),interpolation=cv2.INTER_NEAREST)
    #Nearest for "pixelated" look comparison

    cv2.imshow("Original (Pixelated) vs ESPCN (Sharp)", np.hstack((original_resized, upscaled)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()