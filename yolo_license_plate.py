import torch
from ultralytics import YOLO
import cv2
import os
import numpy as np
import time

LICENSE_PLATE_CLASS_ID = 0  

def load_yolo_v8_model():
    start_time = time.time()  # timing
    model_path = 'C:\\Users\\maran\\Downloads\\yolofinetune\\yolo\\best.pt'
    if not os.path.exists(model_path):
        print(f"Model file does not exist at {model_path}")
        return None
    model = YOLO(model_path)
    print("YOLOv8 model loaded successfully")
    end_time = time.time()  # End timing
    print(f"Model loading time: {end_time - start_time:.4f} seconds")  
    return model

def detect_license_plates_yolo_v8(image, model, confidence_threshold=0.2):
    start_time = time.time()  #  timing
    results = model(image)
    detections = []
    for box in results[0].boxes:
        cls = int(box.cls.item())  
        confidence = box.conf.item()  
        if cls == LICENSE_PLATE_CLASS_ID and confidence > confidence_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert the bounding box to integers
            detections.append(((x1, y1, x2 - x1, y2 - y1), confidence))
    end_time = time.time()  # End timing
    print(f"License plate detection time: {end_time - start_time:.4f} seconds") 
    return detections

def add_watermark(image, text="sompo sigorta", opacity=0.5):
    start_time = time.time()  # Start timing
    # Create a copy of the image to overlay the watermark 
    overlay = image.copy()

    # Set up the font, size, and color 
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (255, 255, 255)  
    font_thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Get the dimensions 
    image_height, image_width = image.shape[:2]

    # Calculate the position for the watermark (center of the image)
    text_x = (image_width - text_size[0]) // 2
    text_y = (image_height + text_size[1]) // 2

    # Add the text to the overlay
    cv2.putText(overlay, text, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    # Blend the overlay with the original image using the specified opacity
    cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)

    end_time = time.time()  # End timing
    print(f"Watermark addition time: {end_time - start_time:.4f} seconds")  
    return image

def cover_license_plate_with_black(image_path, output_path, model, debug=False):
    start_time = time.time()  # timing
    image_load_start_time = time.time()
    
   
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    image_load_end_time = time.time()
    print(f"Image loading time: {image_load_end_time - image_load_start_time:.4f} seconds")

    
    detections = detect_license_plates_yolo_v8(image, model)

    if not detections:
        print(f"No license plates detected in {image_path}.")
        # Add watermark before saving the original image if no license plates are detected
        image = add_watermark(image)
        save_start_time = time.time()
        cv2.imwrite(output_path, image)
        save_end_time = time.time()
        print(f"Image save time: {save_end_time - save_start_time:.4f} seconds")
        return

    if debug:
        for (box, confidence) in detections:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        debug_output_path = output_path.replace('.jpg', '_debug.jpg')
        cv2.imwrite(debug_output_path, image)

    for (box, confidence) in detections:
        x, y, w, h = box
        
        # Cover the license plate with a black rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)

    # Add the watermark after covering the license plate
    image = add_watermark(image)

    save_start_time = time.time()  # Timing image save
    # Save the final image
    cv2.imwrite(output_path, image)
    save_end_time = time.time()
    print(f"Image save time: {save_end_time - save_start_time:.4f} seconds")

    end_time = time.time()  # End timing
    print(f"Total processing time for this image: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python yolo_license_plate.py <input_image_path> <output_image_path> [--debug]")
    else:
        input_image_path = sys.argv[1]
        output_image_path = sys.argv[2]
        debug = '--debug' in sys.argv

        load_start_time = time.time()  
        model = load_yolo_v8_model()
        load_end_time = time.time()  
        print(f"Total model load time: {load_end_time - load_start_time:.4f} seconds")

        if os.path.isfile(input_image_path):
            cover_license_plate_with_black(input_image_path, output_image_path, model, debug)
        else:
            print(f"Error: The input path {input_image_path} is not a valid file.")
