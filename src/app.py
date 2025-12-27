print("✅ app.py started running...")

from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import os
from datetime import datetime
import shutil
from pathlib import Path

app = Flask(__name__)

# ======================
# PATHS
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "weights", "best.pt")
INFER_DIR = os.path.join(BASE_DIR, "..", "models", "predictions", "inference_results")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "predictions")

os.makedirs(INFER_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ======================
# LOAD MODEL
# ======================
model = YOLO(MODEL_PATH)
model.model.names = {0: "emp", 1: "occu"}
print("✅ Model loaded")

# ======================
# GLOBAL STATE
# ======================
latest_data = {
    "empty": 0,
    "occupied": 0,
    "total": 0,
    "last_updated": "-"
}

# ======================
# ROUTES
# ======================
@app.route("/")
def index():
    return render_template("index.html")

# ----------------------
# IMAGE PREDICTION
# ----------------------
@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["image"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    input_name = f"input_{timestamp}.jpg"
    img_path = os.path.join(UPLOAD_DIR, input_name)
    image.save(img_path)

    results = model.predict(
        source=img_path,
        save=True,
        project=INFER_DIR,
        name="web_output",
        conf=0.6,
        iou=0.5,
        show_conf=False
    )

    empty = occupied = 0
    for r in results:
        if r.obb is None:
            continue
        for cls_id in r.obb.cls:
            if int(cls_id) == 0:
                empty += 1
            else:
                occupied += 1

    total = empty + occupied
    time_str = datetime.now().strftime("%d %b %Y, %I:%M %p")

    latest_data.update({
        "empty": empty,
        "occupied": occupied,
        "total": total,
        "last_updated": time_str
    })

    save_dir = Path(results[0].save_dir)
    output_img = list(save_dir.glob("*.jpg"))[0]
    out_name = f"result_{timestamp}.jpg"
    shutil.copy(output_img, os.path.join(UPLOAD_DIR, out_name))

    return render_template(
        "result.html",
        input_image=f"predictions/{input_name}",
        output_image=f"predictions/{out_name}",
        empty=empty,
        occupied=occupied,
        total=total
    )

# ----------------------
# DASHBOARD
# ----------------------
@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        data=latest_data
    )

# ----------------------
# CCTV (DISABLED ON CLOUD)
# ----------------------
@app.route("/cctv")
def cctv():
    return """
    <h2 style='font-family: Arial; text-align:center; margin-top:40px'>
      🎥 Live CCTV works only on local system.<br><br>
      Cloud deployment does not support webcam access.
    </h2>
    """

# ======================
# RUN APP (RENDER SAFE)
# ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
