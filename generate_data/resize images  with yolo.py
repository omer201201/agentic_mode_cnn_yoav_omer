import cv2
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_objects.YOLOv8 import FaceDetector  # Uses your specific detector class
from generate_data_for_gate import letterbox_resize

def prepare_resnet_dataset(input_folder, output_folder, target_size=(224, 224)):
    # Initialize your detector with the safe padding you defined
    detector = FaceDetector(model_path=r"C:\Users\Your0124\final_project\Organized_project\models\yolov8n-face.pt")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')

    print(f"Starting processing: {input_folder} -> {output_folder}")

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(input_folder, filename)
            frame = cv2.imread(img_path)

            if frame is None:
                continue

            # Detect faces using your expansion logic (20% padding)
            results = detector.detect(frame, expand_ratio=0.40)

            for i, data in enumerate(results):
                face_img = data["crop"]

                # Resize to exact ResNet dimensions
                resized_face = letterbox_resize(face_img, target_size)

                # Save the face with a unique name
                save_name = f"crop_{i}_{filename}"
                save_path = os.path.join(output_folder, save_name)
                cv2.imwrite(save_path, face_img)

    print("Processing complete. Faces are ready.")


# Usage
raw_data_path = r"C:\Users\Your0124\final_project\Organized_project\data\system_test\omer\test_yolo"
processed_data_path = r"C:\Users\Your0124\final_project\Organized_project\data\system_test\omer\test_yolo"
prepare_resnet_dataset(raw_data_path, processed_data_path)