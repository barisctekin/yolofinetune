import torch
from ultralytics import YOLO
import cv2
import os
import numpy as np

LICENSE_PLATE_CLASS_ID = 0  

def load_yolo_v8_model():
    model_path = '..\\yolofinetune\\yolo\\best.pt'
    if not os.path.exists(model_path):
        print(f"Model file does not exist at {model_path}")
        return None
    model = YOLO(model_path)
    print("YOLOv8 model loaded successfully")
    return model

def detect_license_plates_yolo_v8(image, model, confidence_threshold=0.2):
    results = model(image)
    print(results[0].boxes)  
    detections = []
    for box in results[0].boxes:
        cls = int(box.cls.item())  
        confidence = box.conf.item()  
        if cls == LICENSE_PLATE_CLASS_ID and confidence > confidence_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert the bounding box to integers
            detections.append(((x1, y1, x2 - x1, y2 - y1), confidence))
    return detections

def blur_license_plate(image_path, output_path, model, debug=False):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Detect license plates 
    detections = detect_license_plates_yolo_v8(image, model)

    if not detections:
        print(f"No license plates detected in {image_path}.")
        # Save the original image if no license plates are detected
        cv2.imwrite(output_path, image)
        return

    if debug:
        for (box, confidence) in detections:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        debug_output_path = output_path.replace('.jpg', '_debug.jpg')
        cv2.imwrite(debug_output_path, image)

    for (box, confidence) in detections:
        x, y, w, h = box
        plate = image[y:y + h, x:x + w]

        if plate.size == 0:
            print(f"Warning: Detected region is empty in {image_path}. Skipping blur.")
            continue

        blurred_plate = cv2.GaussianBlur(plate, (23, 23), 30)
        image[y:y + h, x:x + w] = blurred_plate

    cv2.imwrite(output_path, image)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python yolo_license_plate.py <input_image_path> <output_image_path> [--debug]")
    else:
        input_image_path = sys.argv[1]
        output_image_path = sys.argv[2]
        debug = '--debug' in sys.argv

        model = load_yolo_v8_model()

        if os.path.isfile(input_image_path):
            blur_license_plate(input_image_path, output_image_path, model, debug)
        else:
            print(f"Error: The input path {input_image_path} is not a valid file.")
