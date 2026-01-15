import albumentations as A
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import shutil
import numpy as np

class IndustrialAugmentor:
    def __init__(self):
        # 1. Fixed GaussNoise for newer versions
        # 2. Added min_area and min_visibility to prevent invalid boxes after transform
        self.transform = A.Compose([
            A.MotionBlur(blur_limit=7, p=0.5), 
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3), # Updated syntax
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
        ], bbox_params=A.BboxParams(
            format='yolo', 
            label_fields=['class_labels'],
            min_area=10,        # Drop boxes smaller than 10 pixels
            min_visibility=0.1  # Drop boxes if 90% is cut off by augmentation
        ))

    def sanitize_bboxes(self, bboxes, class_labels):
        """Removes zero-width/height boxes that cause crashes."""
        valid_bboxes = []
        valid_labels = []
        for bbox, label in zip(bboxes, class_labels):
            # YOLO format: [x_center, y_center, width, height]
            # Check if width or height are effectively zero
            if bbox[2] > 0.001 and bbox[3] > 0.001:
                # Clip values to ensure they stay within [0, 1]
                clipped_bbox = [max(0.0, min(1.0, x)) for x in bbox]
                valid_bboxes.append(clipped_bbox)
                valid_labels.append(label)
        return valid_bboxes, valid_labels

    def process_split(self, split_name: str):
        raw_path = Path(f"data/raw/{split_name}")
        processed_path = Path(f"data/processed/{split_name}")
        
        (processed_path / "images").mkdir(parents=True, exist_ok=True)
        (processed_path / "labels").mkdir(parents=True, exist_ok=True)

        print(f"--- Processing {split_name} split ---")
        
        image_files = list((raw_path / "images").glob("*.jpg")) + list((raw_path / "images").glob("*.png"))

        for img_path in tqdm(image_files):
            image = cv2.imread(str(img_path))
            if image is None: continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            label_path = raw_path / "labels" / f"{img_path.stem}.txt"
            if not label_path.exists(): continue
                
            bboxes = []
            class_labels = []
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.split()
                    if len(data) < 5: continue
                    class_labels.append(int(data[0]))
                    bboxes.append([float(x) for x in data[1:]])

            # --- THE FIX: SANITIZE DATA ---
            bboxes, class_labels = self.sanitize_bboxes(bboxes, class_labels)
            
            if not bboxes: # Skip images with no valid objects left
                continue

            try:
                transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                
                # Save Image
                save_img_path = processed_path / "images" / img_path.name
                cv2.imwrite(str(save_img_path), cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
                
                # Save Labels
                save_label_path = processed_path / "labels" / f"{img_path.stem}.txt"
                with open(save_label_path, 'w') as f:
                    for cls, box in zip(transformed['class_labels'], transformed['bboxes']):
                        f.write(f"{cls} {' '.join([str(round(x, 6)) for x in box])}\n")
            except Exception as e:
                # Logging errors instead of crashing is a professional move
                print(f"Skipping {img_path.name} due to error: {e}")

if __name__ == "__main__":
    augmentor = IndustrialAugmentor()
    for split in ['train', 'valid', 'test']:
        augmentor.process_split(split)
    
    # Check if data.yaml exists before copying
    yaml_src = Path("data/raw/pcb_components_roboflow/data.yaml")
    if yaml_src.exists():
        shutil.copy(yaml_src, "data/processed/data.yaml")