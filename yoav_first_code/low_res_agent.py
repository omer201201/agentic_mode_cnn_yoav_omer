import cv2
import os
import numpy as np


class SuperResAgent:
    def __init__(self, model_name="models/FSRCNN-small_x3.pb", scale=3):
        """
        Initializes the Super Resolution Agent (AI Mode Only).
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

            # --- 🛠️ THE FIX IS HERE ---
            # OpenCV only recognizes specific internal names.
            # We check what file you provided and set the internal name accordingly.
            filename_lower = os.path.basename(model_name).lower()
            if 'fsrcnn' in filename_lower:
                algo = 'fsrcnn'
            elif 'edsr' in filename_lower:
                algo = 'edsr'
            elif 'espcn' in filename_lower:
                algo = 'espcn'
            elif 'lapsrn' in filename_lower:
                algo = 'lapsrn'
            else:
                raise ValueError("Unknown model type. Filename must contain fsrcnn, edsr, espcn, or lapsrn.")

            print(f"[SuperRes] Loading file: {os.path.basename(model_name)} as Algorithm: '{algo}'")
            self.sr.readModel(self.model_path)
            # This must be exactly 'fsrcnn', 'edsr', etc.
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

        # 1. Run AI Inference (The Magic Step)
        # The AI dreams up new details to make it 3x bigger.
        result = self.sr.upsample(face_crop)

        # 2. Safety Cap for ResNet (Optional but good practice)
        h, w = result.shape[:2]
        if h > 250 or w > 250:
            # Use INTER_AREA for high-quality shrinking
            result = cv2.resize(result, (224, 224), interpolation=cv2.INTER_AREA)

        return result


def main():
    print("--- 🔍 Testing Super Resolution Agent ---")

    # 1. Initialize Agent
    try:
        # Ensure the path matches exactly where you put the .pb file
        agent = SuperResAgent(model_name="models/FSRCNN-small_x3.pb", scale=3)
    except Exception as e:
        print(e)
        return

    # 2. Load an image
    img_path = "data/gate_dataset/train/low_res/00247.png"
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}")
        return

    original = cv2.imread(img_path)

    # 3. Simulate a "Tiny Input" problem
    # We shrink the image to 1/3rd of its size to create a blurry input
    h, w = original.shape[:2]
    tiny_input = cv2.resize(original, (w // 3, h // 3), interpolation=cv2.INTER_LINEAR)
    print(f"Tiny Input Size: {tiny_input.shape[1]}x{tiny_input.shape[0]}")

    # 4. Run the Agent (AI Upscale)
    upscaled_result = agent.process(tiny_input)
    h_res, w_res = upscaled_result.shape[:2]
    print(f"AI Result Size:  {w_res}x{h_res}")

    # --- 5. Create Single Comparison Figure ---

    # To show them side-by-side, they need the same height.
    # We resize the tiny input UP using standard math (bilinear) to show how blurry it is.
    display_blurry = cv2.resize(tiny_input, (w_res, h_res), interpolation=cv2.INTER_LINEAR)

    # Add labels onto the images so we know which is which
    # (Image, Text, Position, Font, Scale, Color(BGR), Thickness)
    cv2.putText(display_blurry, "Standard Resize (Blurry)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(upscaled_result, "FSRCNN AI (Sharp)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Stack them horizontally
    final_figure = np.hstack((display_blurry, upscaled_result))

    cv2.imshow("Super Resolution Comparison", final_figure)
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()