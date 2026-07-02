import os
import shutil
import random
from glob import glob

def split_dataset(dataset_dir="custom_dataset", split_ratio=0.8):
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    
    # Create train and val directories
    for split in ["train", "val"]:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

    # Get all images that are directly in the images dir
    all_images = [f for f in glob(os.path.join(images_dir, "*.png")) if os.path.isfile(f)]
    random.shuffle(all_images)

    split_index = int(len(all_images) * split_ratio)
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]

    print(f"Moving {len(train_images)} to train, {len(val_images)} to val")

    def move_files(img_paths, split_name):
        for img_path in img_paths:
            base_name = os.path.basename(img_path)
            label_name = base_name.replace(".png", ".txt")
            label_path = os.path.join(labels_dir, label_name)

            # Move image
            shutil.move(img_path, os.path.join(images_dir, split_name, base_name))
            
            # Move label if it exists
            if os.path.exists(label_path):
                shutil.move(label_path, os.path.join(labels_dir, split_name, label_name))

    move_files(train_images, "train")
    move_files(val_images, "val")
    print("Dataset split complete.")

if __name__ == "__main__":
    split_dataset()
