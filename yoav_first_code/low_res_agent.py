import cv2
import os
import numpy as np


class SuperResAgent:
    def __init__(self, model_name="models/FSRCNN-small_x3.pb", scale=3):
        """
        Initializes the Super Resolution Agent
        """
        self.scale = scale
        self.model_path = model_name

        # 1. Verify Model Exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[SuperRes] CRITICAL ERROR: Model file not found at '{self.model_path}'."
            )

        # 2. Load OpenCV Super Resolution
        try:
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()

            # OpenCV only recognizes specific internal names.
            # We check what file provided and set the internal name accordingly.
            filename_lower = os.path.basename(model_name).lower()
            if 'fsrcnn' in filename_lower:
                algo = 'fsrcnn'
            else:
                raise ValueError("Unknown model type. Filename must contain fsrcnn.")

            print(f"[SuperRes] Loading file: {os.path.basename(model_name)} as Algorithm: '{algo}'")
            self.sr.readModel(self.model_path)
            # This must be exactly 'fsrcnn'
            self.sr.setModel(algo, scale)
            print(f"[SuperRes] Success! AI Model loaded (Scale x{scale})")

        except Exception as e:
            raise RuntimeError(f"[SuperRes] Failed to load AI model. Error: {e}")

    def process(self, face_crop):
        """
        Input: Low-res face crop (BGR numpy array).
        Output: High-res AI-upscaled face crop.
        """
        if face_crop is None or face_crop.size == 0:
            return face_crop

        # 1. Run AI Inference new details to make it 3x bigger
        result = self.sr.upsample(face_crop)

        h, w = result.shape[:2]
        print(f"[SuperRes] Upsampling result: {h}x{w}")

        return result


def main():
    print("-- Testing Super Resolution Agent ---")

    # 1. Initialize Agent
    try:
        # Ensure the path matches exactly where you put the .pb file
        agent = SuperResAgent(model_name="models/FSRCNN-small_x3.pb", scale=3)
    except Exception as e:
        print(e)
        return

    # 2. Load an image
    img_path = r"C:\Users\Your0124\pycharm_project_test\data\train\yoav\IMG_1622.jpg"
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}")
        return

    original = cv2.imread(img_path)

    print(f"Tiny Input Size: {original.shape[1]}x{original.shape[0]}")

    # --- 4. Run the Agent (AI Upscale) ---
    upscaled_reday = agent.process(original)

    # ---Resize the original to match the AI result height ---
    original_ready = cv2.resize(original, (900, 900), interpolation=cv2.INTER_AREA)
    upscaled_result =cv2.resize(upscaled_reday, (900, 900), interpolation=cv2.INTER_AREA)

    # Add labels onto the images
    cv2.putText(original_ready, "Original (Resized)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(upscaled_result, "FSRCNN AI (Sharp)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Stack them horizontally (Heights now match perfectly)
    final_figure = np.hstack((original_ready, upscaled_result))

    cv2.imshow("Super Resolution Comparison", final_figure)
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
