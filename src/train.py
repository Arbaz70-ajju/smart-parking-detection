# src/train.py

import yaml
from ultralytics import YOLO
from data_utils import verify_dataset
from model_utils import save_model
import os


def main(config_path=r"C:\OSI_Parking\models\yolov11s-obb\args.yaml"):
    """
    Main training script for YOLO OBB model.
    Loads training config, verifies dataset and trains the model.
    """

    # ----- Load Configuration -----
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")

    print(f"📂 Loading configuration from: {config_path}")

    with open(config_path, "r") as f:
        args = yaml.safe_load(f)

    # ----- Verify Dataset -----
    data_yaml = verify_dataset(args["data"])  # returns correct absolute path

    # ----- Load Model -----
    model_path = args["model"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ YOLO model not found: {model_path}")

    print(f"✅ Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    # ----- Training -----
    print("🚀 Starting YOLO training...")
    results = model.train(
        data=data_yaml,
        epochs=args.get("epochs", 70),
        imgsz=args.get("imgsz", 640),
        batch=args.get("batch", 8),
        project=args.get("project", "runs/train"),
        name=args.get("name", "parking_yolo"),
        exist_ok=True,
        device=args.get("device", 0),
        workers=args.get("workers", 8),
        verbose=args.get("verbose", True),
    )

    # ----- Save Trained Model -----
    save_path = "models/finetuned.pt"
    save_model(model, save_path)

    print("✅ Training complete!")
    print(f"📦 Model saved at: {save_path}")


if __name__ == "__main__":
    main()
