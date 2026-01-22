import cv2
import os
import numpy as np


class SuperResAgent:
    def __init__(self, model_name="models/FSRCNN_x3.pb", scale=3):
        """
        Initializes the Super Resolution Agent (AI Mode Only).

        Args:
            model_name (str): Filename of the .pb model (must be in 'models/' folder).
            scale (int): Upscaling factor (2, 3, or 4).
        """
        self.scale = scale
        self.model_path = model_name

        # 1. Verify Model Exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[SuperRes] CRITICAL ERROR: Model file not found at '{self.model_path}'. "
                f"You strictly requested AI mode. Please download {model_name}."
            )

        # 2. Load OpenCV Super Resolution
        try:
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            filename = os.path.basename(model_name)
            # Extract algorithm name (e.g., 'fsrcnn' from 'FSRCNN_x3.pb')
            # This is case-sensitive for OpenCV
            algo = filename.split('_')[0].lower()

            self.sr.readModel(self.model_path)
            self.sr.setModel(algo, scale)
            print(f"[SuperRes]  Loaded AI Model: {model_name} (Scale x{scale})")

        except Exception as e:
            raise RuntimeError(f"[SuperRes] Failed to load AI model. Error: {e}")

    def process(self, face_crop):
        """
        Input: Low-res face crop (BGR numpy array).
        Output: High-res AI-upscaled face crop.
        """
        # Safety: If you accidentally send nothing, send nothing back (don't crash)
        if face_crop is None or face_crop.size == 0:
            return face_crop

        # 1. Run AI Inference (THE MAGIC STEP)
        # self.sr is the engine we built in __init__.
        # .upsample() takes the small image and uses the Neural Network
        # to dream up new pixels, making it 3x bigger and sharper.
        result = self.sr.upsample(face_crop)

        # 2. Safety Cap for ResNet
        # If the input was 100x100, the result is now 300x300.
        # That is too big (slows down the system). ResNet only needs ~224px.
        # So if it's huge, we shrink it back down to a standard size.
        h, w = result.shape[:2]
        if h > 250 or w > 250:
            result = cv2.resize(result, (224, 224), interpolation=cv2.INTER_AREA)

        return result


def main():
    print("--- 🔍 Testing Super Resolution Agent ---")

    # 1. Initialize Agent
    # Make sure the .pb file is in the 'models' folder!
    agent = SuperResAgent()

    # 2. Load an image
    # Use one of your images
    img_path = "data/gate_dataset/low_res/face_1.jpg"

    if not os.path.exists(img_path):
        print("Image not found.")
        return

    original = cv2.imread(img_path)

    # 3. Simulate a "Low Res" problem
    # We shrink the image to 1/3rd of its size
    h, w = original.shape[:2]
    low_res_input = cv2.resize(original, (w // 3, h // 3), interpolation=cv2.INTER_LINEAR)

    print(f"Original Size: {w}x{h}")
    print(f"Low Res Input: {low_res_input.shape[1]}x{low_res_input.shape[0]}")

    # 4. Run the Agent (The Magic)
    upscaled_result = agent.process(low_res_input)

    print(f"Result Size:   {upscaled_result.shape[1]}x{upscaled_result.shape[0]}")

    # 5. Visual Comparison
    # Let's resize the low-res input back up cleanly just for visual comparison (Bilinear)
    # This shows what "dumb" zooming looks like vs AI zooming.
    dumb_zoom = cv2.resize(low_res_input, (upscaled_result.shape[1], upscaled_result.shape[0]))

    cv2.imshow("1. Low Res Input (Tiny)", low_res_input)
    cv2.imshow("2. Standard Zoom (Blurry)", dumb_zoom)
    cv2.imshow("3. AI Super Res (Sharp)", upscaled_result)

    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()