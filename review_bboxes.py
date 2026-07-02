import cv2
import os

def draw_yolo_boxes(image_path, label_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load {image_path}")
        return
        
    h, w, _ = img.shape
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center, y_center = float(parts[1]) * w, float(parts[2]) * h
                    box_w, box_h = float(parts[3]) * w, float(parts[4]) * h
                    
                    x1 = int(x_center - box_w / 2)
                    y1 = int(y_center - box_h / 2)
                    x2 = int(x_center + box_w / 2)
                    y2 = int(y_center + box_h / 2)
                    
                    # Draw green bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
    cv2.imwrite(output_path, img)
    print(f"Saved review image to {output_path}")

if __name__ == "__main__":
    img_dir = "dataset/images"
    lbl_dir = "dataset/labels"
    out_dir = "dataset/review"
    
    os.makedirs(out_dir, exist_ok=True)
    
    for filename in os.listdir(img_dir):
        if filename.endswith(".png"):
            base = filename.replace(".png", "")
            img_path = os.path.join(img_dir, filename)
            lbl_path = os.path.join(lbl_dir, f"{base}.txt")
            out_path = os.path.join(out_dir, filename)
            
            draw_yolo_boxes(img_path, lbl_path, out_path)
