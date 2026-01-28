import cv2
import os
import numpy as np
import random
from pathlib import Path

# CONFIG
SOURCE_DIR = r"C:\Users\Your0124\pycharm_project_test\data\gate_dataset\normal"  # Put your good images here first!
ROOT_DIR = r"C:\Users\Your0124\pycharm_project_test\data\gate_dataset"
TARGET_SIZE = (128, 128)  # Standardize for the Gate CNN

def create_folders():
    classes = ["low_light", "motion_blur", "low_res"]
    for c in classes:
        os.makedirs(os.path.join(ROOT_DIR, c), exist_ok=True)


def letterbox_resize(img, target_size=TARGET_SIZE, color=(0, 0, 0)):
    """Resizes image while maintaining aspect ratio and adding padding."""
    h, w = img.shape[:2]
    # Calculate the ratio to fit the image into the target size
    r = min(target_size[0] / h, target_size[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))

    # Resize keeping the proportions
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Calculate padding needed to reach target_size
    dw = target_size[0] - new_unpad[0]
    dh = target_size[1] - new_unpad[1]

    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)

    # Add black bars
    return cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)

def make_low_light(img):
    # 1. Reduce brightness (Gamma Correction)
    # 2.5 to 3.5 is very dark. Try 1.5 to 2.0 if it's too dark to see anything.
    gamma = random.uniform(1.5, 2.5)
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    dark = cv2.LUT(img, look_up_table)

    # Randomize ISO grain/noise intensity
    noise_level = random.randint(5, 25)
    noise = np.zeros(dark.shape, np.int16) # Use int16 to allow negative values temporarily
    cv2.randn(noise, 0, noise_level) # Generate random normal noise

    # 3. Combine and Clip
    # This ensures values stay between 0-255
    noisy_dark = cv2.add(dark.astype(np.int16), noise)
    final_img = np.clip(noisy_dark, 0, 255).astype(np.uint8)

    return final_img


def make_motion_blur(img):
    """Simulates movement at any random angle (0-360 degrees)."""
    kernel_size = random.choice([7, 9, 11, 13])
    angle = random.uniform(0, 360)

    # Create the horizontal kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

    # Rotate the kernel to the random angle
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

    kernel /= np.sum(kernel)  # Normalize to maintain brightness
    final_img = cv2.filter2D(img, -1, kernel)

    return final_img


def make_low_res(img):
    # Downscale then Upscale to simulate pixelation
    h, w = img.shape[:2]
    scale = random.uniform(0.1, 0.25)  # Resize to 10-25% of original
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    # Resize back to original so the CNN can read it
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated


def generate():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source folder {SOURCE_DIR} does not exist.")
        return

    create_folders()
    images = list(Path(SOURCE_DIR).glob("*/*.jpg")) + list(Path(SOURCE_DIR).glob("*.jpg"))
    print(f"Found {len(images)} normal images. Generating bad versions...")

    for img_path in images:
        filename = img_path.name
        img = cv2.imread(str(img_path))

        if img is None: continue
        padded_img = letterbox_resize(img)

        # --- STEP 2: Save Classes ---
        cv2.imwrite(os.path.join(ROOT_DIR, "normal", filename), padded_img)
        cv2.imwrite(os.path.join(ROOT_DIR, "low_light", filename), make_low_light(padded_img))
        cv2.imwrite(os.path.join(ROOT_DIR, "motion_blur", filename), make_motion_blur(padded_img))
        cv2.imwrite(os.path.join(ROOT_DIR, "low_res", filename), make_low_res(padded_img))

    print(" Data Generation Complete! Check your data folders.")


if __name__ == "__main__":

    generate()
