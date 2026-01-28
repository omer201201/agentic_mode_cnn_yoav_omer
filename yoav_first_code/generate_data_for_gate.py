import cv2
import os
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
# We point SOURCE_DIR to your new downloaded folder
SOURCE_DIR = r"C:\Users\Your0124\pycharm_project_test\generate_data\FFHQ_128X128"
ROOT_DIR = r"C:\Users\Your0124\pycharm_project_test\data\gate_dataset"
TARGET_SIZE = (128, 128)
NUM_IMAGES_TO_USE = 2000  # How many faces to process
TRAIN_RATIO = 0.8  # 80% train, 20% validation

def create_gate_folders():
    """Creates the nested train/val folder structure."""
    for split in ["train", "val"]:
        for c in ["normal", "low_light", "motion_blur", "low_res"]:
            path = os.path.join(ROOT_DIR, split, c)
            os.makedirs(path, exist_ok=True)


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
        print(f"Error: Could not find {SOURCE_DIR}")
        return

    create_gate_folders()

    # Get list of images and shuffle them for a random sample
    all_images = list(Path(SOURCE_DIR).glob("*.png")) + list(Path(SOURCE_DIR).glob("*.jpg"))
    random.shuffle(all_images)
    selected_images = all_images[:NUM_IMAGES_TO_USE]

    split_point = int(len(selected_images) * TRAIN_RATIO)

    print(f"--  Generating Data from FFHQ Thumbs ---")

    for idx, img_path in enumerate(tqdm(selected_images)):
        current_split = "train" if idx < split_point else "val"
        img = cv2.imread(str(img_path))
        if img is None: continue

        # Standardize
        padded = letterbox_resize(img)
        filename = img_path.name

        base_path = os.path.join(ROOT_DIR, current_split)

        # Save the 4 classes
        cv2.imwrite(os.path.join(base_path, "normal", filename), padded)
        cv2.imwrite(os.path.join(base_path, "low_light", filename), make_low_light(padded))
        cv2.imwrite(os.path.join(base_path, "motion_blur", filename), make_motion_blur(padded))
        cv2.imwrite(os.path.join(base_path, "low_res", filename), make_low_res(padded))

    print(f" Success! Folders ready at: {ROOT_DIR}")


if __name__ == "__main__":
    generate()
