# src/data_utils.py

import os

def verify_dataset(data_dir="data"):
    """
    Ensures that YOLO dataset folders exist and are structured properly.
    """
    expected = ["images", "labels"]
    for split in ["train", "val", "test"]:
        for folder in expected:
            path = os.path.join(data_dir, split, folder)
            if not os.path.exists(path):
                print(f"⚠️ Missing folder: {path}")
            else:
                print(f"✅ Found: {path}")
    yaml_path = os.path.join(data_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"❌ data.yaml not found in {data_dir}")
    print(f"✅ Dataset verified at: {data_dir}")
    return yaml_path
