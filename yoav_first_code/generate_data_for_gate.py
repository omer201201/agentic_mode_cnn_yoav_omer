import cv2
import os
import numpy as np
import random
from pathlib import Path

# CONFIG
SOURCE_DIR = "data/gate_dataset/normal"  # Put your good images here first!
ROOT_DIR = "data/gate_dataset"


def create_folders():
    classes = ["low_light", "motion_blur", "low_res"]
    for c in classes:
        os.makedirs(os.path.join(ROOT_DIR, c), exist_ok=True)


def make_low_light(img):
    # Reduce brightness significantly (Gamma Correction)
    gamma = random.uniform(2.5, 3.5)  # Higher gamma = darker
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    dark = cv2.LUT(img, look_up_table)
    # Add some ISO noise (grain) which happens in dark
    noise = np.random.normal(0, 5, dark.shape).astype(np.uint8)
    return cv2.add(dark, noise)


def make_motion_blur(img):
    # Simulate camera shake
    size = random.choice([7, 9, 11])  # Kernel size
    kernel_motion_blur = np.zeros((size, size))
    # Horizontal blur
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    return cv2.filter2D(img, -1, kernel_motion_blur)


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