import cv2
import numpy as np
import os

class LowLightAgent:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Initializes the Low Light Enhancement Agent.
        We use CLAHE: It boosts contrast locally in small grid tiles.

        params:
        - clip_limit: Threshold for contrast limiting. Higher = more contrast (and more noise).
        - tile_grid_size: Size of grid for histogram equalization. Input image will be divided into these tiles.
        """
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def process(self, image):
        """
        Input: Dark BGR image (numpy array).
        Output: Enhanced BGR image.
        """
        if image is None:
            return None

        # 1. Convert BGR to LAB color space
        # L = Lightness (Intensity)
        # A = Green-Red
        # B = Blue-Yellow
        # We process ONLY the L channel to avoid messing up the colors.
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # 2. Split the channels
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # 3. Apply CLAHE to the L-channel (Lightness)
        # This brightens the dark parts intelligently
        enhanced_l = self.clahe.apply(l_channel)

        # 4. Merge the channels back together
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))

        # 5. Convert back to BGR so OpenCV can display it
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_bgr

def main():
        print("--- 🌑 Testing Low Light Agent ---")

        # 1. Initialize the Agent
        # You can tweak 'clip_limit' (Try 2.0, 3.0, or 4.0) to see what looks best.
        agent = LowLightAgent(clip_limit=3.0)

        # 2. Load a dark image
        # CHANGE THIS PATH to one of your real dark images from the 'yoav' folder
        img_path = "data/gate_dataset/low_light/face_1.jpg"

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