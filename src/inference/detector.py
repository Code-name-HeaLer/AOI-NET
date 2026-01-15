import cv2
from ultralytics import YOLO
import yaml
from pathlib import Path
import sys

# Add root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.spatial_math import SpatialSynchronizer

class AOIInferenceEngine:
    def __init__(self, model_path, config_path):
        self.model = YOLO(model_path)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sync = SpatialSynchronizer(self.config)
        
        # Convert config targets to lowercase for easier matching
        self.target_classes = [c.lower() for c in self.config['detection_logic']['target_classes']]
        self.min_conf = self.config['detection_logic']['min_confidence']

    def run_live(self, source=0):
        # source=0 is usually built-in webcam. Try 1 if using an external USB cam.
        cap = cv2.VideoCapture(source)
        
        # Performance optimization: Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print(f"--- AOI Inference Engine Active ---")
        print(f"Tracking Classes: {self.target_classes}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Inference
            results = self.model.predict(frame, conf=self.min_conf, stream=True, verbose=False)

            for r in results:
                for box in r.boxes:
                    # Get box data
                    b = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0])
                    name = self.model.names[cls_id]
                    conf = float(box.conf[0])

                    # VETERAN LOGIC: Case-insensitive matching
                    if name.lower() in self.target_classes or "resestor" in name.lower():
                        # Math
                        centroid_x, _ = self.sync.get_centroid(b)
                        delay = self.sync.calculate_trigger_delay(centroid_x)

                        # Drawing
                        color = (0, 255, 0) # Green
                        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
                        
                        label = f"{name} {conf:.2f} | Delay: {delay:.2f}s"
                        cv2.putText(frame, label, (int(b[0]), int(b[1]-10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("Industrial AOI-Net Pipeline", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    BEST_MODEL = "runs/detect/train/weights/best.pt"
    CONFIG = "config/hardware_config.yaml"
    engine = AOIInferenceEngine(BEST_MODEL, CONFIG)
    engine.run_live()