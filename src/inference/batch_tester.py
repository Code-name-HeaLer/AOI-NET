import cv2
import os
from ultralytics import YOLO
import yaml
from pathlib import Path
from tqdm import tqdm
import sys

# Add root to path for math utils
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.spatial_math import SpatialSynchronizer

class AOIBatchTester:
    def __init__(self, model_path, config_path, output_dir="data/test_results"):
        self.model = YOLO(model_path)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sync = SpatialSynchronizer(self.config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mapping targets
        self.target_classes = [c.lower() for c in self.config['detection_logic']['target_classes']]
        self.min_conf = self.config['detection_logic']['min_confidence']

    def process_test_folder(self, test_images_path):
        test_path = Path(test_images_path)
        images = list(test_path.glob("*.jpg")) + list(test_path.glob("*.png"))
        
        print(f"--- Starting Offline Validation on {len(images)} images ---")
        
        for img_path in tqdm(images):
            frame = cv2.imread(str(img_path))
            if frame is None: continue

            # Run Inference
            results = self.model.predict(frame, conf=self.min_conf, verbose=False)

            for r in results:
                for box in r.boxes:
                    b = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0])
                    name = self.model.names[cls_id]
                    conf = float(box.conf[0])

                    # Case-insensitive check
                    if name.lower() in self.target_classes or "resestor" in name.lower():
                        # Use our math to show what the delay WOULD have been
                        centroid_x, _ = self.sync.get_centroid(b)
                        delay = self.sync.calculate_trigger_delay(centroid_x)

                        # Drawing logic
                        color = (255, 0, 0) # Blue for test results
                        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
                        
                        label = f"{name} {conf:.2f} | Delay: {delay:.2f}s"
                        cv2.putText(frame, label, (int(b[0]), int(b[1]-10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Save the result to the output directory
            output_path = self.output_dir / img_path.name
            cv2.imwrite(str(output_path), frame)

        print(f"\n--- Validation Complete ---")
        print(f"Results saved to: {self.output_dir.absolute()}")

if __name__ == "__main__":
    # 1. Update these paths to match your folder structure
    BEST_MODEL = "runs/detect/train/weights/best.pt"
    CONFIG = "config/hardware_config.yaml"
    TEST_DATA = "data/raw/test/images"
    
    tester = AOIBatchTester(BEST_MODEL, CONFIG)
    tester.process_test_folder(TEST_DATA)