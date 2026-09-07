import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import random
import uuid
import shutil

# Configuration
RAW_DIR = "raw_gun_images"
DATASET_DIR = "dataset"
IMAGES_TRAIN_DIR = os.path.join(DATASET_DIR, "images", "train")
IMAGES_VAL_DIR = os.path.join(DATASET_DIR, "images", "val")
LABELS_TRAIN_DIR = os.path.join(DATASET_DIR, "labels", "train")
LABELS_VAL_DIR = os.path.join(DATASET_DIR, "labels", "val")

# Ensure directories exist
for d in [IMAGES_TRAIN_DIR, IMAGES_VAL_DIR, LABELS_TRAIN_DIR, LABELS_VAL_DIR]:
    os.makedirs(d, exist_ok=True)

# Initialize MediaPipe HandLandmarker in IMAGE mode
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)

count = 0
skipped = 0

print(f"Scanning '{RAW_DIR}' folder for images...")

with vision.HandLandmarker.create_from_options(options) as landmarker:
    for filename in os.listdir(RAW_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            raw_img_path = os.path.join(RAW_DIR, filename)
            img = cv2.imread(raw_img_path)
            
            if img is None:
                continue
                
            # Convert to MediaPipe format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Detect hand
            result = landmarker.detect(mp_image)
            
            if result.hand_landmarks:
                # Find Bounding Box
                hand_landmarks = result.hand_landmarks[0]
                h, w, _ = img.shape
                x_coords = [lm.x for lm in hand_landmarks]
                y_coords = [lm.y for lm in hand_landmarks]
                
                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)
                
                # Add 10% padding
                padding_x = (xmax - xmin) * 0.1
                padding_y = (ymax - ymin) * 0.1
                xmin = max(0.0, xmin - padding_x)
                ymin = max(0.0, ymin - padding_y)
                xmax = min(1.0, xmax + padding_x)
                ymax = min(1.0, ymax + padding_y)
                
                # 20% chance to go to validation set, 80% to train
                split = "val" if random.random() < 0.2 else "train"
                
                img_dir = IMAGES_VAL_DIR if split == "val" else IMAGES_TRAIN_DIR
                lbl_dir = LABELS_VAL_DIR if split == "val" else LABELS_TRAIN_DIR
                
                file_id = str(uuid.uuid4())
                img_dest_path = os.path.join(img_dir, f"{file_id}.jpg")
                lbl_dest_path = os.path.join(lbl_dir, f"{file_id}.txt")
                
                # Copy image to YOLO dataset
                shutil.copy(raw_img_path, img_dest_path)
                
                # Save YOLO format label
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin
                
                with open(lbl_dest_path, "w") as f:
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                count += 1
            else:
                skipped += 1
                print(f"Skipped {filename}: No hand detected.")

print(f"==========================================")
print(f"Processing Complete!")
print(f"Successfully auto-labeled and formatted: {count} images.")
if skipped > 0:
    print(f"Skipped {skipped} images where no hand was visible.")
print(f"You can now run 'python train_yolo.py'")
print(f"==========================================")
