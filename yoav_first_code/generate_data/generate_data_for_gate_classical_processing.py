import os
import cv2
import shutil
import numpy as np
import random

# --- ⚙ CONFIGURATION ---
INPUT_FOLDER = r"C:\Users\Your0124\pycharm_project_test\results"
OUTPUT_BASE = r"C:\Users\Your0124\pycharm_project_test\data\gate_dataset"

# --- 🧪 IDEAL PROBLEM ANCHORS (The 'Perfect' Bad Image) ---
# We define what 100% bad looks like for each category
IDEAL_LOW_LIGHT = 10  # Avg Luminance of 10 is '100% low light'
IDEAL_BLUR = 10  # Laplacian Var of 10 is '100% blurred'
IDEAL_LOW_RES = 30  # Size of 25px is '100% low res'

# Normalization Thresholds (Anything better than this is 0% bad)
MIN_LIGHT = 110
MIN_SHARP = 110
MIN_SIZE = 70


def get_best_fit_label(img):
    h, w = img.shape[:2]

    # 1. HARD SKIP: Garbage Filter
    if h < 30 or w < 30:
        return "skip"

    # 2. Calculate Raw Metrics
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    lum = np.mean(yuv[:, :, 0])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()

    size = min(h, w)

    # 3. Calculate "Problem Scores" (0.0 to 1.0)
    # Higher score = more likely to be that category
    score_light = np.clip((MIN_LIGHT - lum) / (MIN_LIGHT - IDEAL_LOW_LIGHT), 0, 1)
    score_blur = np.clip((MIN_SHARP - var) / (MIN_SHARP - IDEAL_BLUR), 0, 1)
    score_res = np.clip((MIN_SIZE - size) / (MIN_SIZE - IDEAL_LOW_RES), 0, 1)

    scores = {
        "low_light": score_light,
        "motion_blur": score_blur,
        "low_res": score_res
    }

    # 4. Find the Winner
    best_cat = max(scores, key=scores.get)
    max_score = scores[best_cat]

    # 5. Threshold for "Normal"
    # If even the worst problem isn't very bad, it's a Normal image.
    if max_score < 0.35:
        return "normal"

    return best_cat


def main():
    # Setup folders
    categories = ["normal", "low_light", "low_res", "motion_blur"]
    for split in ["train", "valid"]:
        for cat in categories:
            os.makedirs(os.path.join(OUTPUT_BASE, split, cat), exist_ok=True)

    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg'))]
    random.shuffle(files)

    sorted_data = {cat: [] for cat in categories}

    for f in files:
        img = cv2.imread(os.path.join(INPUT_FOLDER, f))
        if img is None: continue

        label = get_best_fit_label(img)
        if label != "skip":
            sorted_data[label].append(os.path.join(INPUT_FOLDER, f))

    # Perform 80/20 split copy
    for cat, paths in sorted_data.items():
        split_idx = int(len(paths) * 0.8)
        for i, path in enumerate(paths):
            target_split = "train" if i < split_idx else "valid"
            shutil.copy2(path, os.path.join(OUTPUT_BASE, target_split, cat, os.path.basename(path)))
        print(f"Sorted {cat}: {len(paths)} images")


if __name__ == "__main__":
    main()