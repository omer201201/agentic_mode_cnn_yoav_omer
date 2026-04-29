import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ----------------------------------------
# 1. Face Detector Agent Definition
# PURPOSE: This is the very first stage of the pipeline. It scans the raw
# camera frame, locates all human faces, and extracts them into perfectly
# square "crops" so the downstream models (Gate and ResNet) get standardized data.
# ----------------------------------------
# --- CONFIGURATION ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = {
    "Yolo": os.path.join(PROJECT_ROOT, "models", "yolov8n-face.pt")
}

class FaceDetector:
    def __init__(self, model_path=MODEL_PATH["Yolo"], conf_threshold=0.5):

        #Initialize the YOLOv8 Face detector.
        # 1.Auto-select GPU (CUDA) for Jetson
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[YOLO] Loading Face Model on {self.device}...")

        # 2. Load the specific Face Model
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"[Error] Could not find model at {model_path}")
            raise e
        # The minimum confidence score (0.0 to 1.0) required to consider a "face"
        self.conf_threshold = conf_threshold

    def detect(self, frame, expand_ratio=0.40):

        # 1. Run Inference (verbose=False keeps the terminal clean)
        results = self.model(frame, verbose=False)
        detections = []
        height_img, width_img, _ = frame.shape

        # results[0].boxes contains all the detections for this specific frame
        for box in results[0].boxes:
            conf = float(box.conf[0].item())

            if conf >= self.conf_threshold:
                # 1. Get original tight coordinates
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()

                # 1. Get original dimensions and center
                box_w = x2 - x1
                box_h = y2 - y1
                center_x = x1 + box_w // 2
                center_y = y1 + box_h // 2

                # 2. Determine the "Square Side"
                # we use the max dimension so we don't chop off the chin or forehead
                # We apply the expand_ratio here: (1 + 0.x) makes it x% larger
                side = int(max(box_w, box_h) * (1 + expand_ratio))
                half_side = side // 2

                # 3. Apply expansion from the CENTER
                x1 = max(0, center_x - half_side)
                y1 = max(0, center_y - half_side)
                x2 = min(width_img, center_x + half_side)
                y2 = min(height_img, center_y + half_side)

                # 3. CROP THE FACE
                #copy() is important so we don't modify the original frame
                face_crop = frame[y1:y2, x1:x2].copy()

                # 4. Save everything
                detections.append({
                    "crop": face_crop,  #The image for the Gate
                    "coords": (x1, y1, x2, y2),  #Location for drawing
                    "conf": conf
                })

        return detections



# --- Helper Function for Camera Setup ---
def get_camera(width=1280, height=720):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Could not open camera.")
        return None

    # Force C270 to HD Mode
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

# ----------------------------------------
# 3. Testing Functions
# ----------------------------------------

def Yolo_with_camera(model_path):
    #Boots up the webcam and runs real-time face detection.
    #Press 'q' in the window to exit the loop.
    print("--- Testing Face Detector with Live Camera ---")
    try:
        detector = FaceDetector(model_path=model_path)
        cap = get_camera()

        if cap is None:
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Detect faces
            faces = detector.detect(frame)

            # 2. Draw bounding boxes
            for (x1, y1, x2, y2, conf) in faces:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Face: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 3. Display live feed
            cv2.imshow("YOLO Live (Press 'q' to quit)", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"[Camera Error] {e}")


def Yolo_with_image(image_path, model_path):

    #Loads a static image from the hard drive, detects faces,
    #and pops up windows showing both the full frame and isolated crops.

    print(f"--- Testing Face Detector on Static Image ---")
    print(f"Loading: {image_path}")

    try:
        detector = FaceDetector(model_path=model_path)

        frame = cv2.imread(image_path)
        if frame is None:
            print("Error: Image not found. Check your file path.")
            return

        # 1. Run Detection
        results = detector.detect(frame)
        print(f"Found {len(results)} faces.")

        # 2. Display Loop
        for i, data in enumerate(results):
            face_img = data["crop"]
            coords = data["coords"]

            # Show the isolated square crop
            cv2.imshow(f"Face_Crop_{i}", face_img)

            # Draw targeting box on the main image
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)

        # Show the final annotated master frame
        cv2.imshow("Full Frame", frame)

        print("Press any key in the image window to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"[Image Error] {e}")


# ----------------------------------------
# 4. Main Execution Block
# ----------------------------------------
if __name__ == "__main__":
    # --- CONFIGURATION ---
    TEST_IMAGE = r"C:\Users\Your0124\pycharm_project_test\data\resnet_dataset\test1\2d4c4ced-2ade-4a7d-8853-b1718cef1020.JPG"

    # --- EXECUTION SWITCH ---
    # Camera/Image selection

    # Yolo_with_camera(model_path=MODEL_PATH["Yolo"])
    Yolo_with_image(image_path=TEST_IMAGE, model_path=MODEL_PATH["Yolo"])