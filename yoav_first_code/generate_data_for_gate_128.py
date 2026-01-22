import cv2
import numpy as np
import os
import random
import shutil

# --- ⚙️ CONFIGURATION ---
SOURCE_NORMAL_DIR = r"C:\Users\Omon0415\pycharm_projects\Final_Project_yoav&omer\yoav_first_code\data\gate_dataset\normal"
OUTPUT_DIR = "data/gate_train_ready"
NUM_IMAGES_TO_PROCESS = 2000
IMG_SIZE = 128  # Target Square Size


# --- 🛠️ HELPER FUNCTIONS ---

def resize_with_padding(img, target_size=128):
    """
    Resizes an image to target_size x target_size while KEEPING aspect ratio.
    Adds black borders (padding) to make it square.
    """
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)

    # New dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image content
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a black square canvas
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # Center the image on the canvas
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def create_low_light(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    factor = random.uniform(0.50, 0.65)
    v = (v.astype(np.float32) * factor).astype(np.uint8)
    final = cv2.merge((h, s, v))
    return cv2.cvtColor(final, cv2.COLOR_HSV2BGR)


def create_motion_blur(img):
    kernel_size = random.choice([7, 9, 11, 13])
    kernel = np.zeros((kernel_size, kernel_size))
    if random.random() > 0.5:
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    else:
        kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(img, -1, kernel)


def create_low_res(img):
    # Since we are already 128x128, we shrink to 24x24 and back
    h, w = img.shape[:2]
    small = cv2.resize(img, (24, 24), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def main():
    if not os.path.exists(SOURCE_NORMAL_DIR):
        print(f"❌ Error: Could not find source folder: {SOURCE_NORMAL_DIR}")
        return

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    classes = ['normal', 'low_light', 'motion_blur', 'low_res']
    for split in ['train', 'val']:
        for c in classes:
            path = os.path.join(OUTPUT_DIR, split, c)
            os.makedirs(path, exist_ok=True)

    files = [f for f in os.listdir(SOURCE_NORMAL_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(files)

    if NUM_IMAGES_TO_PROCESS is not None:
        files = files[:NUM_IMAGES_TO_PROCESS]

    print(f"🚀 Processing {len(files)} images with PADDING (Letterbox)...")

    for i, filename in enumerate(files):
        src_path = os.path.join(SOURCE_NORMAL_DIR, filename)
        img = cv2.imread(src_path)
        if img is None: continue

        # --- KEY CHANGE IS HERE ---
        # Instead of simple cv2.resize, we use the padding function
        img = resize_with_padding(img, target_size=IMG_SIZE)

        split = 'train' if i < len(files) * 0.8 else 'val'

        # Save Normal
        cv2.imwrite(os.path.join(OUTPUT_DIR, split, 'normal', filename), img)

        # Generate Bad Versions (They will now also have padding, which is good!)
        cv2.imwrite(os.path.join(OUTPUT_DIR, split, 'low_light', filename), create_low_light(img))
        cv2.imwrite(os.path.join(OUTPUT_DIR, split, 'motion_blur', filename), create_motion_blur(img))
        cv2.imwrite(os.path.join(OUTPUT_DIR, split, 'low_res', filename), create_low_res(img))

        if i % 100 == 0:
            print(f"   Processed {i} / {len(files)}...", end='\r')

    print(f"\n✅ SUCCESS! Images saved to {OUTPUT_DIR}")
    print("   All images are now exactly 128x128 with black borders.")


if __name__ == "__main__":
    main()