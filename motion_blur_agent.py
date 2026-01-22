import cv2
import numpy as np
import os

class MotionBlurAgent:
    def __init__(self, strength=1):
        """
        Initializes the Deblurring Agent.
        We use an 'Unsharp Mask' technique via a Sharpening Kernel.
        """
        # A standard sharpening kernel
        # The center value (9) boosts the pixel, while neighbors (-1) reduce surrounding noise.
        # You can make it stronger by increasing the center value slightly.
        self.kernel = np.array([[-1, -1, -1],
                                [-1, 9, -1],
                                [-1, -1, -1]])

    def process(self, face_crop):
        """
        Input: Blurry face crop.
        Output: Sharpened face crop.
        """
        if face_crop is None or face_crop.size == 0:
            return face_crop

        # Apply the sharpening filter
        # -1 means the output image will have the same depth as the source.
        sharpened = cv2.filter2D(face_crop, -1, self.kernel)

        return sharpened


def main():
    print("--- 💨 Testing Motion Blur Agent ---")

    # 1. Initialize the Agent
    agent = MotionBlurAgent()

    # 2. Load a NORMAL image (We will blur it ourselves to test)
    # Use one of your normal yoav images
    img_path = "data/gate_dataset/motion_blur/face_1.jpg"

    if not os.path.exists(img_path):
        print(f"❌ Error: Image not found at {img_path}")
        return

    original = cv2.imread(img_path)

    # 3. Simulate Motion Blur (Create the problem)
    # We use a 15x15 averaging kernel to make it look blurry
    blur_kernel_size = 15
    blurred_img = cv2.blur(original, (blur_kernel_size, blur_kernel_size))

    # 4. Run the Agent (Fix the problem)
    result = agent.process(blurred_img)

    # 5. Visual Comparison
    print("Displaying results... (Press any key to close)")

    # Resize for easier viewing
    h, w = original.shape[:2]
    target_width = 400
    scale = target_width / w

    img1 = cv2.resize(original, None, fx=scale, fy=scale)
    img2 = cv2.resize(blurred_img, None, fx=scale, fy=scale)
    img3 = cv2.resize(result, None, fx=scale, fy=scale)

    # Label the images
    cv2.putText(img1, "1. Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img2, "2. Simulated Blur", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img3, "3. Agent Fixed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Stack them side-by-side
    comparison = np.hstack((img1, img2, img3))

    cv2.imshow("Motion Blur Test Pipeline", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()