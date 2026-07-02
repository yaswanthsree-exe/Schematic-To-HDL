import cv2
import numpy as np
import os
import glob
import re

DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

def parse_yolo_labels(filepath, img_width, img_height):
    """Reads YOLO bounding boxes and returns coordinates (x, y, w, h)."""
    boxes = []
    if not os.path.exists(filepath):
        return boxes
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                # YOLO format: class_id x_center y_center width height
                class_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
                
                # Convert normalized to absolute
                abs_w = int(w * img_width)
                abs_h = int(h * img_height)
                abs_x = int(xc * img_width - abs_w / 2)
                abs_y = int(yc * img_height - abs_h / 2)
                
                boxes.append((abs_x, abs_y, abs_w, abs_h))
    return boxes

def trace_wires(image_path, debug=True):
    # 1. Load image and labels
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
        
    h_img, w_img = img.shape[:2]
    
    # Extract file ID from path to find the matching label
    match = re.search(r'schem_([^.]+)\.png', os.path.basename(image_path))
    if not match: return
    file_id = match.group(1)
    label_path = os.path.join(LABELS_DIR, f"schem_{file_id}.txt")
    
    boxes = parse_yolo_labels(label_path, w_img, h_img)
    
    # 2. Wire Isolation
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV) # Black lines become white
    
    # Mask out the bounding boxes
    wire_mask = thresh.copy()
    for (x, y, w, h) in boxes:
        # Increase box size slightly to ensure terminals are cleanly cut
        pad = 5
        cv2.rectangle(wire_mask, (max(0, x-pad), max(0, y-pad)), 
                       (min(w_img, x+w+pad), min(h_img, y+h+pad)), 
                       0, -1) # Fill black to mask out the gate
                       
    # 3. Line Detection
    # Morphological skeletonization could be good, but we can try HoughLinesP first
    lines = cv2.HoughLinesP(wire_mask, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
    
    if debug:
        debug_img = img.copy()
        # Draw gates in green
        for (x, y, w, h) in boxes:
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        # Draw identified lines in blue
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
        # Save output
        out_path = f"dataset/review/traced_{file_id}.png"
        cv2.imwrite(out_path, debug_img)
        print(f"Saved trace debug to {out_path}")

if __name__ == "__main__":
    os.makedirs("dataset/review", exist_ok=True)
    images = glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    for img_path in images[:2]:
        print(f"Tracing wires for {img_path}...")
        trace_wires(img_path)
