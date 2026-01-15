import os
from ultralytics import YOLO, settings
from pathlib import Path

# --- VETERAN FIX: Disable MLflow via Global Settings ---
# This updates the actual YOLO configuration file on your machine
settings.update({'mlflow': False})

# Additional environment overrides to be 100% sure
os.environ["MLFLOW_TRACKING_URI"] = "" 
os.environ["REPORTTO"] = "none"

def train_model():
    # 1. Path Management (Absolute paths are safest on Windows)
    dataset_config = Path("data/raw/data.yaml").absolute()
    
    # 2. Load Model (Nano version for high speed/low VRAM)
    model = YOLO('yolov8n.pt') 

    # 3. Training Parameters
    # We remove any "mlflow" keys here as they are not officially supported args
    train_params = {
        "data": str(dataset_config),
        "epochs": 50,
        "imgsz": 640,
        "batch": 4,            # Reduced to 4 to be safe given the memory error
        "device": 0,
        "optimizer": 'AdamW',
        "plots": True,         
        "overlap_mask": False, 
        "cache": False,
        "workers": 0,          # SET TO 0: This solves the "Paging File" error on Windows
        "classes": [0, 1, 2, 3, 7, 8] 
    }

    print("--- Starting GPU-Optimized Industrial Training ---")
    print("Note: MLflow has been globally disabled to prevent Windows URI errors.")
    
    try:
        # Start training
        model.train(**train_params)
        
        # 4. Export for Production (The final deliverable)
        print("\nTraining complete. Exporting model to ONNX...")
        onnx_path = model.export(format="onnx")
        print(f"Success! Production model saved at: {onnx_path}")
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        print("\nTroubleshooting Tip:")
        print("If you still see MLflow errors, run 'pip uninstall mlflow' in your venv.")

if __name__ == "__main__":
    train_model()