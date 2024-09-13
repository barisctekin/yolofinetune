import os
import time
import cv2
from ultralytics import YOLO
from pathlib import Path


current_dir = Path(__file__).resolve().parent
image_dir = current_dir / 'images'
output_dir = current_dir / 'output'


output_dir.mkdir(exist_ok=True)

LICENSE_PLATE_CLASS_ID = 0  

def load_yolo_v8_model():
    start_time = time.time()  
    model_path = current_dir / 'yolo' / 'best.pt'
    if not model_path.exists():
        print(f"Model file does not exist at {model_path}")
        return None
    model = YOLO(str(model_path))  # Convert Path object to string
    print("YOLOv8 model loaded successfully")
    end_time = time.time()  
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
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert the bounding box 
            detections.append(((x1, y1, x2 - x1, y2 - y1), confidence))
    end_time = time.time()  # End timing
    print(f"License plate detection time: {end_time - start_time:.4f} seconds") 
    return detections

def add_watermark(image, text="sompo sigorta", opacity=0.5):
    start_time = time.time()  # Start timing
    # Create a copy of the image 
    overlay = image.copy()

    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (255, 255, 255)  
    font_thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Get the dimensions 
    image_height, image_width = image.shape[:2]

    # Calculate the position 
    text_x = (image_width - text_size[0]) // 2
    text_y = (image_height + text_size[1]) // 2

    
    cv2.putText(overlay, text, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    
    cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)

    end_time = time.time()  # End timing
    print(f"Watermark addition time: {end_time - start_time:.4f} seconds")  
    return image

def cover_license_plate_with_black(image_path, output_path, model, debug=False):
    start_time = time.time()  # timing
    image_load_start_time = time.time()
    
    # Load 
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    image_load_end_time = time.time()
    print(f"Image loading time: {image_load_end_time - image_load_start_time:.4f} seconds")

    # Detect license plates 
    detections = detect_license_plates_yolo_v8(image, model)

    if not detections:
        print(f"No license plates detected in {image_path}.")
        # Add watermark before 
        image = add_watermark(image)
        save_start_time = time.time()
        cv2.imwrite(str(output_path), image)
        save_end_time = time.time()
        print(f"Image save time: {save_end_time - save_start_time:.4f} seconds")
        return

    if debug:
        for (box, confidence) in detections:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # No saving of debug images

    for (box, confidence) in detections:
        x, y, w, h = box
        
        # Cover the license plate with  black 
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)

    # Add the watermark after covering the license plate
    image = add_watermark(image)

    save_start_time = time.time()  # Timing image save
    # Save the final image
    cv2.imwrite(str(output_path), image)
    save_end_time = time.time()
    print(f"Image save time: {save_end_time - save_start_time:.4f} seconds")

    end_time = time.time()  # End timing
    print(f"Total processing time for this image: {end_time - start_time:.4f} seconds")

# Main function to process all images 
def process_images_in_directory(image_dir, output_dir, model, debug=False):
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    files = os.listdir(image_dir)

    for file in files:
        file_path = image_dir / file
        output_file_path = output_dir / file
        print(f"Processing {file_path}...")
        cover_license_plate_with_black(file_path, output_file_path, model, debug)

if __name__ == "__main__":
    model = load_yolo_v8_model()
    if model:
        process_images_in_directory(image_dir, output_dir, model, debug=True)
