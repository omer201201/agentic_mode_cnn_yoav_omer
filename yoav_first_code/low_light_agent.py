import os
import cv2
import numpy as np


class DynamicLowLightAgent:
    def __init__(self):
        pass

    def _get_dynamic_params(self, l_channel):
        avg_brightness = np.mean(l_channel)

        # We've softened these to prevent the "washed out" look
        if avg_brightness < 40:
            clip_limit, gamma, denoise = 2.0, 0.8, 7
        elif avg_brightness < 80:
            clip_limit, gamma, denoise = 1.2, 0.9, 4
        else:
            clip_limit, gamma, denoise = 0.8, 1.0, 0

        return clip_limit, gamma, denoise

    def process(self, image):
        if image is None: return None

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clip, gamma, denoise = self._get_dynamic_params(l)

        # 1. Subtle CLAHE to avoid "ugly" grain
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)

        # 2. Merge back to BGR
        merged = cv2.merge((enhanced_l, a, b))
        enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # 3. Smoother Gamma (avoids the ghostly skin tone)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        final_bgr = cv2.LUT(enhanced_bgr, table)

        # 4. Light Denoising - specifically for the 'face grain'
        if denoise > 0:
            final_bgr = cv2.fastNlMeansDenoisingColored(final_bgr, None, denoise, denoise, 7, 21)

        return final_bgr

def main():
        print("--- 🌑 Testing Low Light Agent ---")

        # 1. Initialize the Agent
        # You can tweak 'clip_limit' (Try 2.0, 3.0, or 4.0) to see what looks best.
        agent = DynamicLowLightAgent()

        # 2. Load a dark image
        # CHANGE THIS PATH to one of your real dark images from the 'yoav' folder
        img_path = "data/gate_dataset/train/low_light/00257.png"

        if not os.path.exists(img_path):
            print(f" Error: Image not found at {img_path}")
            return

        original = cv2.imread(img_path)

        # 3. Run the Agent
        result = agent.process(original)

        # 4. Show Side-by-Side Comparison
        # We stack them horizontally (Left: Original, Right: Fixed)
        comparison = np.hstack((original, result))

        # Optional: Resize if the image is too big for your screen
        h, w = comparison.shape[:2]
        if w > 1500:
            scale = 1500 / w
            comparison = cv2.resize(comparison, None, fx=scale, fy=scale)

        cv2.imshow("Left: Original (Dark) | Right: Agent Result (Enhanced)", comparison)

        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()