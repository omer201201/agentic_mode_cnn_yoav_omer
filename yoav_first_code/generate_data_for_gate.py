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


def make_motion_blur(img):
    # 1. Choose a random size (length of motion) and angle
    size = random.choice([7, 9, 11, 13])
    angle = random.uniform(0, 360)  # Any angle between 0 and 360 degrees

    # 2. Create the base basic horizontal kernel
    # (Just like the previous version)
    kernel = np.zeros((size, size))
    center = int((size - 1) / 2)
    kernel[center, :] = 1

    # 3. Rotate the kernel to the random angle
    # Calculate the rotation matrix around the center point
    M = cv2.getRotationMatrix2D((center, center), angle, 1.0)

    # Apply the rotation to the kernel itself
    # We use INTER_LINEAR to smooth the line edges when rotating
    rotated_kernel = cv2.warpAffine(kernel, M, (size, size), flags=cv2.INTER_LINEAR)

    # 4. Normalize the rotated kernel
    # Crucial: After rotation and interpolation, the sum is no longer exactly 'size'.
    # We must divide by the new sum to ensure image brightness doesn't change.
    rotated_kernel = rotated_kernel / rotated_kernel.sum()

    # 5. Apply filter
    return cv2.filter2D(img, -1, rotated_kernel)


def make_low_res(img):
    # Downscale then Upscale to simulate pixelation
    h, w = img.shape[:2]
    scale = random.uniform(0.3, 0.4)  # Resize to 30-40% of original
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    # Resize back to original so the CNN can read it
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated


def generate():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source folder {SOURCE_DIR} does not exist.")
        return

    create_folders()
    images = list(Path(SOURCE_DIR).glob("*.jpg"))
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
        lowers = make_low_res(img)
        cv2.imwrite(f"{ROOT_DIR}/low_res/{filename}", lowers)

    print(" Data Generation Complete! Check your data folders.")


if __name__ == "__main__":

    generate()
