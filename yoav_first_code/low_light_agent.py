import os
import cv2
import numpy as np


class DynamicLowLightAgent:
    def __init__(self):
        pass

    def _get_dynamic_params(self, l_channel):
        avg_brightness = np.mean(l_channel)

        # handel for different dark images
        print("avg_brightness:", avg_brightness) # test to see the avg_brightness
        if avg_brightness < 20:
            clip_limit, gamma, denoise = 6.0, 0.80, 10
        elif avg_brightness < 30:
            clip_limit, gamma, denoise = 5.0, 0.85, 10
        elif avg_brightness < 40:
            clip_limit, gamma, denoise =  3.0, 0.85, 7
        elif avg_brightness < 80:
            clip_limit, gamma, denoise = 2, 1.0, 2
        else:
            clip_limit, gamma, denoise = 0, 1.0, 0

        return clip_limit, gamma, denoise

    def process(self, image):

        if image is None: return None
        #LAB - format for allows the agent to isolate the L (Lightness) channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clip, gamma, denoise = self._get_dynamic_params(l)

        # --- NEW: Early Exit Logic ---
        if clip == 0 and gamma == 1.0 and denoise == 0:
            return image  # Return the original BGR image without any math
        # -----------------------------
        # 1. Subtle CLAHE to avoid "ugly" grain

        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(16, 16))
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
        print("---  Testing Low Light Agent ---")

        # 1. Initialize the Agent
        agent = DynamicLowLightAgent()

        # 2. Load a dark image
        img_path = r"C:\Users\Your0124\pycharm_project_test\data\resnet_dataset\test\3.jpeg"

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
