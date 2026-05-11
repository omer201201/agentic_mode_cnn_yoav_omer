import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm

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

        print(f"Found {len(image_files)} images. Starting Motion Blur batch process...")

        for filename in tqdm(image_files, desc="Deblurring Images"):
            img_path = os.path.join(input_dir, filename)

            img = cv2.imread(img_path)
            if img is None:
                continue

            try:
                # Apply deblurring
                fixed_img = self.process(img)

                # Save the enhanced image
                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, fixed_img)
            except Exception as e:
                print(f"\nError processing {filename}: {e}")

        print(f"\nFinished processing! All images saved to: {output_dir}")


def main():
    print("--- Testing Motion Blur Agent ---")

    # 1. Initialize the Agent
    agent = MotionBlurAgent()

    print("\n--- Running Batch Process ---")

    input_folder = r"C:\Users\Your0124\pycharm_project_test\agentic_mode_cnn_yoav_omer-Organized_Project_12-04-2026\agentic_mode_cnn_yoav_omer-Organized_Project_12-04-2026\Organized_project_13_4\generate_data\other"
    output_folder = r"C:\Users\Your0124\pycharm_project_test\agentic_mode_cnn_yoav_omer-Organized_Project_12-04-2026\agentic_mode_cnn_yoav_omer-Organized_Project_12-04-2026\Organized_project_13_4\generate_data\other"

    agent.process_directory(input_folder, output_folder)


if __name__ == "__main__":
    main()