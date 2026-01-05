import json
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO


# ---------- 1. Build same classifier model as in training ----------

def build_id_model(num_classes=3, weights_path="models/id_classifier_resnet18.pt"):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------- 2. Preprocess face crop for the classifier ----------

def get_preprocess_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ---------- 3. Load YOLO face detector ----------

def load_yolo_model():
    # use face-trained YOLO if you have one; otherwise generic YOLO and treat detected person area as face
    # example: "yolov8n-face.pt" or "yolov8n.pt"
    model = YOLO("yolov8n.pt")  # change to your face model if available
    return model


# ---------- 4. Inference loop ----------

def main():
    # 1. Load class mapping
    with open("models/class_mapping.json", "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # 2. Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    id_model = build_id_model(num_classes=len(class_to_idx))
    id_model.to(device)

    yolo_model = load_yolo_model()
    preprocess = get_preprocess_transform()

    # 3. Open camera
    cap = cv2.VideoCapture(0)  # adjust index or use RTSP/CSI pipeline

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting real-time face identification (Yoav / Omer / Unknown)... Press 'q' to quit.")

    CONF_THRESHOLD = 0.5
    ID_THRESHOLD = 0.7  # min prob to accept Yoav/Omer, otherwise Unknown

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # 4. Run YOLO inference
        results = yolo_model(frame, verbose=False)

        # We assume first result (batch size 1)
        detections = results[0].boxes

        for box in detections:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            # If using generic YOLO, often class 0 is "person"
            # You might want to filter only person / face class
            # Here we assume we treat any detection as candidate face:
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = box.xyxy[0].int().tolist()

            # Clip coordinates to frame size
            h, w, _ = frame.shape
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            # 5. Crop face region
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            # BGR -> RGB
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            # 6. Preprocess and run ID classifier
            face_tensor = preprocess(face_rgb).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]

            with torch.no_grad():
                logits = id_model(face_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                best_prob, best_idx = torch.max(probs, dim=0)

            best_prob = float(best_prob.item())
            best_idx = int(best_idx.item())
            label_name = idx_to_class[best_idx]

            # 7. Apply ID threshold – if confidence low, mark as unknown
            if best_prob < ID_THRESHOLD or label_name == "other":
                display_label = "Unknown"
            else:
                # translate internal class name to nice display
                if label_name.lower() == "yoav":
                    display_label = "Yoav"
                elif label_name.lower() == "omer":
                    display_label = "Omer"
                else:
                    display_label = "Unknown"

            # 8. Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{display_label} ({best_prob:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 9. Show result frame
        cv2.imshow("Face ID - Yoav / Omer / Unknown", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
