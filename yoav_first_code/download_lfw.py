import os
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

# CONFIG
# Where to save the "Good" images
SAVE_DIR = "data/gate_dataset/normal"
NUM_IMAGES_TO_SAVE = 500


def get_faces():
    print(" Downloading/Loading LFW dataset... (this might take a minute)")

    # This automatically downloads the data if you don't have it
    # min_faces_per_person=20 ensures we get clear, distinct faces
    lfw_people = fetch_lfw_people(min_faces_per_person=1, resize=None)  # resize=None keeps original quality

    print(f" Loaded {len(lfw_people.images)} faces.")
    return lfw_people.images


def save_faces(images):
    # Create directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Clear existing files in 'normal' so we don't duplicate
    for f in os.listdir(SAVE_DIR):
        os.remove(os.path.join(SAVE_DIR, f))

    count = 0
    print(f" Saving {NUM_IMAGES_TO_SAVE} images to {SAVE_DIR}...")

    # Indices to pick random images
    indices = np.arange(len(images))
    np.random.shuffle(indices)

    for i in indices:
        if count >= NUM_IMAGES_TO_SAVE:
            break

        # LFW images are normalized (0-1 float). Convert to 0-255 uint8 for OpenCV.
        img = images[i]

        # Normalize to 0-255 range
        img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        # LFW is Black and White by default in sklearn (usually).
        # We convert to BGR so your code handles 3 channels (Color) correctly.
        img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

        # Save
        filename = os.path.join(SAVE_DIR, f"face_{count}.jpg")
        cv2.imwrite(filename, img_color)
        count += 1

    print(f" Done! Saved {count} images.")


if __name__ == "__main__":
    faces = get_faces()
    save_faces(faces)
