# src/model_utils.py

import os
from ultralytics import YOLO

def load_model(model_path="models/weights/best.pt"):
    """
    Loads a YOLO model from a given file path.
    Ensures the model file exists before loading.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at: {model_path}")

    print(f"✅ Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    return model


def save_model(model, save_path="models/finetuned.pt"):
    """
    Saves a YOLO model after training.
    Ultralytics uses export() for saving weights.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"💾 Exporting trained model to: {save_path}")
    model.export(format="pt", path=save_path)

    print(f"✅ Model saved successfully at: {save_path}")
