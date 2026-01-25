import cv2
import numpy as np
import os

class MotionBlurAgent:
    def __init__(self, amount=1.5, threshold=0):
        """
        Initializes the Deblurring Agent using a Gaussian Unsharp Mask.
        This is more natural than a standard kernel for facial features.
        """
        self.amount = amount
        self.threshold = threshold

    def process(self, face_crop):
        if face_crop is None or face_crop.size == 0:
            return face_crop

        # 1. Create a blurred version of the image (Gaussian)
        # This acts as a low-pass filter
        blurred = cv2.GaussianBlur(face_crop, (5, 5), 1.0)

        # 2. Subtract the blur from the original to get the 'details'
        # sharpened = original + (original - blurred) * amount
        sharpened = cv2.addWeighted(face_crop, 1.0 + self.amount, blurred, -self.amount, 0)

        return sharpened

def main():
    print("--- 💨 Testing Motion Blur Agent on Real Data ---")

    # 1. Initialize the Agent
    agent = MotionBlurAgent(amount=1.2)

    # 2. Path to your REAL blurry image
    img_path = "data/gate_dataset/train/motion_blur/00297.png"

    if not os.path.exists(img_path):
        print(f"❌ Error: Image not found at {img_path}")
        return

    # 3. Load the real blurry image
    blurry_img = cv2.imread(img_path)
    if blurry_img is None: return

    # 4. Run the Agent (Fix the real blur)
    result = agent.process(blurry_img)

    # 5. Visual Comparison (Real Blur vs Fixed)
    h, w = blurry_img.shape[:2]
    # Resize only for display purposes
    display_blurry = cv2.resize(blurry_img, (224, 224))
    display_fixed = cv2.resize(result, (224, 224))

    # Add Labels
    cv2.putText(display_blurry, "Real Blur", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(display_fixed, "Agent Fixed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Stack Side-by-Side
    comparison = np.hstack((display_blurry, display_fixed))

    print("✅ Displaying results... (Press any key to close)")
    cv2.imshow("Real Motion Blur Correction", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()