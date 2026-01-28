import cv2
import os
import numpy as np
import random
from pathlib import Path

# CONFIG
SOURCE_DIR = "data/gate_dataset/normal"  # Put your good images here first!
ROOT_DIR = r"C:\Users\Your0124\pycharm_project_test\data\gate_dataset"


def create_folders():
    classes = ["low_light", "motion_blur", "low_res"]
    for c in classes:
        os.makedirs(os.path.join(ROOT_DIR, c), exist_ok=True)


def make_low_light(img):
    # 1. Reduce brightness (Gamma Correction)
    # 2.5 to 3.5 is very dark. Try 1.5 to 2.0 if it's too dark to see anything.
    gamma = random.uniform(1.5, 2.5)
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    dark = cv2.LUT(img, look_up_table)

    # 2. Add realistic ISO noise
    # We create noise, then use cv2.add weighted or clip it so it doesn't "wrap"
    noise = np.zeros(dark.shape, np.int16)  # Use int16 to allow negative values temporarily
    cv2.randn(noise, 0, 10)  # Generate random normal noise

    # 3. Combine and Clip
    # This ensures values stay between 0-255
    noisy_dark = cv2.add(dark.astype(np.int16), noise)
    final_img = np.clip(noisy_dark, 0, 255).astype(np.uint8)

    return final_img


def create_motion_blur(img):
    kernel_size = random.choice([7, 9, 11, 13])
    kernel = np.zeros((kernel_size, kernel_size))
    if random.random() > 0.5:
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    else:
        kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(img, -1, kernel)

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

        # 1. Generate Low Light
        dark = make_low_light(img)
        cv2.imwrite(f"{ROOT_DIR}/low_light/{filename}", dark)

        # 2. Generate Blur
        blur = make_motion_blur(img)
        cv2.imwrite(f"{ROOT_DIR}/motion_blur/{filename}", blur)

        # 3. Generate Low Res
        lowres = make_low_res(img)
        cv2.imwrite(f"{ROOT_DIR}/low_res/{filename}", lowres)

    print("✅ Data Generation Complete! Check your data folders.")


if __name__ == "__main__":

    generate()
