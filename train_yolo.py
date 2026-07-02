from ultralytics import YOLO
import os

def train():
    # Use absolute path to the Roboflow dataset
    data_yaml = os.path.join(os.path.dirname(__file__), "final_dataset_hdl", "data.yaml")
    
    print(f"Dataset config: {data_yaml}")
    print("Initializing YOLOv8 model for training...")
    
    # Load pre-trained YOLOv8-nano
    model = YOLO("yolov8n.pt")

    # Train with proper settings for 1.2k image dataset
    print("Starting training on Roboflow logic gates dataset (1212 images, 7 classes)...")
    results = model.train(
        data=data_yaml,
        epochs=50,            # 50 epochs for solid convergence
        imgsz=640,            # Images are already 640x640
        project="runs",
        name="roboflow_gates",
        batch=16,
        patience=10,          # Early stopping if no improvement for 10 epochs
        lr0=0.01,             # Initial learning rate
        lrf=0.01,             # Final learning rate factor
        mosaic=1.0,           # Mosaic augmentation
        flipud=0.5,           # Vertical flip augmentation
        fliplr=0.5,           # Horizontal flip augmentation  
        degrees=10,           # Rotation augmentation
        translate=0.1,        # Translation augmentation
        scale=0.5,            # Scale augmentation
        hsv_h=0.015,          # Hue augmentation
        hsv_s=0.7,            # Saturation augmentation
        hsv_v=0.4,            # Value augmentation
        workers=4,
        verbose=True,
    )
    
    print(f"\nTraining complete!")
    print(f"Best weights: {results.save_dir}/weights/best.pt")
    print(f"Results:  mAP50={results.results_dict.get('metrics/mAP50(B)', 'N/A')}")

if __name__ == "__main__":
    train()
