import cv2
import torch
from ultralytics import YOLO
import time

class FaceDetector:
    def __init__(self, model_path="models/yolov8n-face.pt", conf_threshold=0.5):
        """
        Initialize the YOLOv8 Face detector.
        """
        # 1. Device Setup: Auto-select GPU (CUDA) for Jetson
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[YOLO] Loading Face Model on {self.device}...")

        # 2. Load the specific Face Model
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"[Error] Could not find model at {model_path}. Please check the path.")
            raise e

        self.conf_threshold = conf_threshold

    def detect(self, frame, expand_ratio=0.20):
        """
        Input: Single image frame
        Output: List of tuples (x1, y1, x2, y2, confidence)
        expand_ratio: How much to expand the box (0.20 = 20% bigger)
        """
        # Run inference
        results = self.model(frame, verbose=False)

        detections = []
        height_img, width_img, _ = frame.shape

        for box in results[0].boxes:
            conf = float(box.conf[0].item())

            if conf >= self.conf_threshold:
                # 1. Get original tight coordinates
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()

                # 2. Calculate the expansion amount
                box_width = x2 - x1
                box_height = y2 - y1

                # Expand horizontally and vertically
                x_pad = int(box_width * expand_ratio)
                y_pad = int(box_height * expand_ratio)

                # 3. Apply expansion (and ensure we don't go outside the image)
                x1 = max(0, x1 - x_pad)
                y1 = max(0, y1 - y_pad)
                x2 = min(width_img, x2 + x_pad)
                y2 = min(height_img, y2 + y_pad)

                # 3. CROP THE FACE
                # Note: copy() is important so we don't modify the original frame
                face_crop = frame[y1:y2, x1:x2].copy()

                # 4. Save everything
                detections.append({
                    "crop": face_crop,  # The image for the Gate
                    "coords": (x1, y1, x2, y2),  # Location for drawing
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

'''
# --- Main Block for Testing ---
if __name__ == "__main__":
    print("Testing Face Detector...")

    # NOTE: Ensure 'yolov8n-face.pt' is in the 'models' folder,
    # or update the path below to where you saved it.
    try:
        detector = FaceDetector(model_path="models/yolov8n-face.pt")
        cap = get_camera()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # DETECT
            faces = detector.detect(frame)

            # DRAW
            for (x1, y1, x2, y2, conf) in faces:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Face: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("YOLO Face Debug", frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
'''
# --- Main Block for Testing with an Image ---
# --- Test Block ---
if __name__ == "__main__":
    print("Testing Face Detector + Cropping...")

    try:
        # Ensure path is correct
        detector = FaceDetector(model_path="models/yolov8n-face.pt")

        # Load test image
        frame = cv2.imread("test_image.JPG")
        if frame is None:
            print("Error: test_face.jpg not found.")
            exit()

        # Run Detection
        results = detector.detect(frame)
        print(f"Found {len(results)} faces.")

        for i, data in enumerate(results):
            # 1. Get the data
            face_img = data["crop"]
            coords = data["coords"]

            # 2. Show the CROPPED face (What the Gate will see)
            cv2.imshow(f"Face_Crop_{i}", face_img)

            # 3. Draw on original frame (Visualization)
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)

        # Show original frame with boxes
        cv2.imshow("Full Frame", frame)

        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(e)
