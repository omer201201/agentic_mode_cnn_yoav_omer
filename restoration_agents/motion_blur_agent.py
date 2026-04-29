import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ----------------------------------------
# 1. Motion Blur Agent Definition
# PURPOSE: This agent acts as a filter in the pipeline. It mathematicaly
# evaluates how "sharp" a face crop is. If the image is blurry, it dynamically
# calculates how much sharpening to apply based on the severity of the blur.
# ----------------------------------------
class MotionBlurAgent:
    def __init__(self, blur_threshold=250.0):
        #Initializes the Deblurring Agent using a Gaussian Unsharp Mask.
        self.blur_threshold = blur_threshold  # Lower = blurrier

    def get_blur_score(self, face_crop):

        #Calculates a numerical score representing how sharp the image is.
        #Higher score = Sharper edges. Lower score = Blurry.

        # Convert to grayscale because edge detection relies on light intensity, not color.
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        # --- THE LAPLACIAN VARIANCE ---
        # 1. The Laplacian operator calculates the 2nd derivative of the image pixels.
        #    It highlights areas where the color changes rapidly (edges).
        # 2. .var() calculates the mathematical variance of these highlighted edges.
        #    If an image is sharp, there are many strong edges (high variance).
        #    If an image is blurry, the edges are smooth and spread out (low variance).

        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def process(self, face_crop):
        #The main execution block. It assesses the image and applies a fix if necessary.
        if face_crop is None or face_crop.size == 0:
            return face_crop

        # 1. Measure the blur score
        score = self.get_blur_score(face_crop)
        #print(f"Blur Score: {score:.2f}")

        # 2. Check if fix is needed
        if score < self.blur_threshold:
            # DYNAMIC CALCULATION:
            # We don't want to apply the same heavy sharpening to every image.
            # We want 'amount' to increase as 'score' decreases.

            # Example 1: If threshold is 250 and score is 50, raw_amount = (250/50)*0.5 = 2.5 (Heavy fix)
            # Example 2: If threshold is 250 and score is 200, raw_amount = (250/200)*0.5 = 0.62 (Light fix)
            raw_amount = (self.blur_threshold / max(score, 1.0)) * 0.5
            dynamic_amount = min(raw_amount, 2.5)
            #print(f"Applying Dynamic Sharpness: {dynamic_amount:.2f}")

            # 3a. Create a slightly blurred version of the original image.
            blurred = cv2.GaussianBlur(face_crop, (5, 5), 1.0)

            # 3b. cv2.addWeighted blends two images together.
            # The math happening here is: Sharpened = Original + (Original - Blurred) * Amount
            # By subtracting the blurred version from the original, we are left with ONLY the edges.
            # We then add those edges back into the original image, effectively "boosting" the sharpness.
            sharpened = cv2.addWeighted(face_crop, 1.0 + dynamic_amount, blurred, -dynamic_amount, 0)

            return sharpened
        # If the score was higher than the threshold, return the original image untouched.
        return face_crop

def main():
    print("---  Testing Motion Blur Agent on Real Data ---")

    # 1. Initialize the Agent
    agent = MotionBlurAgent()

    # 2. Path to the REAL blurry image
    img_path = r"C:\Users\Your0124\final_project\Organized_project\data\system_test\yoav\motion_blur\IMG_8317.jpg"

    if not os.path.exists(img_path):
        print(f" Error: Image not found at {img_path}")
        return

    # 3. Load the real blurry image
    blurry_img = cv2.imread(img_path)
    if blurry_img is None: return

    # 4. Run the Agent (Fix the blur)
    result = agent.process(blurry_img)

    # 5. Visual Comparison (Real Blur vs Fixed)
    h, w = blurry_img.shape[:2]
    # Resize only for display purposes
    display_blurry = cv2.resize(blurry_img, (224, 224))
    display_fixed = cv2.resize(result, (224, 224))

    # Add Labels
    cv2.putText(display_blurry, "before", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(display_fixed, "Agent Fixed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Stack Side-by-Side
    comparison = np.hstack((display_blurry, display_fixed))

    print(" Displaying results... (Press any key to close)")
    cv2.imshow("Motion Blur Correction", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()