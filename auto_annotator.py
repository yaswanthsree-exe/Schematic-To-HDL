import cv2
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import glob
import shutil

# Paths
MODEL_PATH = "../logic_gate_recognizer/backend/model.pth"
CLASSES_PATH = "../logic_gate_recognizer/backend/classes.txt"
INPUT_DIR = "internet_images"
OUTPUT_IMAGES_DIR = "internet_dataset/images/train"
OUTPUT_LABELS_DIR = "internet_dataset/labels/train"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_recognizer():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(f"Could not find model at {MODEL_PATH} or {CLASSES_PATH}")
        
    with open(CLASSES_PATH, "r") as f:
        classes = f.read().splitlines()
        
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, classes

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def annotate_images():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
    
    print("Loading Logic Gate Recognizer Model...")
    try:
        model, classes = load_recognizer()
        print(f"Loaded {len(classes)} classes: {classes}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.*"))
    if not image_paths:
        print(f"No images found in {INPUT_DIR}. Please drop your downloaded schematics there!")
        return
        
    for path in image_paths:
        filename = os.path.basename(path)
        img_name, ext = os.path.splitext(filename)
        if ext.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
            
        print(f"Processing {filename}...")
        img = cv2.imread(path)
        if img is None: continue
        
        h_img, w_img = img.shape[:2]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Target internet schematics which are usually dark lines on white backgrounds.
        # Invert so symbols are white on black, making cv2.findContours happy.
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yolo_labels = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # Filter huge boundary boxes or tiny specs of dust
            if area < 150 or area > (h_img * w_img * 0.4):
                continue
                
            pad = 5
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)
            
            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
                
            conf_val = conf.item()
            class_id = pred.item()
            
            # Threshold to ignore text or random lines that get contoured
            if conf_val > 0.85:
                # Calculate YOLO normalized coordinates
                xc = (x + w/2) / w_img
                yc = (y + h/2) / h_img
                norm_w = w / w_img
                norm_h = h / h_img
                
                yolo_labels.append(f"{class_id} {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}")
                
        if yolo_labels:
            out_img_path = os.path.join(OUTPUT_IMAGES_DIR, filename)
            shutil.copy(path, out_img_path)
            
            out_txt_path = os.path.join(OUTPUT_LABELS_DIR, f"{img_name}.txt")
            with open(out_txt_path, "w") as f:
                f.write("\n".join(yolo_labels))
                
    print(f"Finished auto-annotating! Results saved to {OUTPUT_IMAGES_DIR} and {OUTPUT_LABELS_DIR}")

if __name__ == "__main__":
    annotate_images()
