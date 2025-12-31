import os
import uuid
import time

from flask import Flask, render_template, request

import cv2
import torch
from ultralytics import YOLO

from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ================= Flask App =================
app = Flask(__name__)

# ================= Paths =================
BASE = os.path.dirname(os.path.abspath(__file__))

UPLOAD = os.path.join(BASE, "static/uploads")
RESULT = os.path.join(BASE, "static/results")
MODELS = os.path.join(BASE, "models")

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(RESULT, exist_ok=True)

YOLO_PATH = os.path.join(MODELS, "best_yolov8_model_1.pt")
FASTER_PATH = os.path.join(MODELS, "best_faster_rcnn_resnet50_fpn.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= Class Names (KITTI) =================
CLASS_NAMES = {
    1: "Car",
    2: "Pedestrian",
    3: "Cyclist"
}

# ================= Load YOLO =================
print("Loading YOLOv8...")
yolo = YOLO(YOLO_PATH)

# ================= Load Faster R-CNN =================
print("Loading Faster R-CNN...")

def load_faster():
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES) + 1)

    ckpt = torch.load(FASTER_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    model.to(device)
    model.eval()
    return model

faster = load_faster()

# ================= Route =================
@app.route("/", methods=["GET", "POST"])
def index():
    data = {}

    if request.method == "POST":
        conf = float(request.form.get("conf", 0.25))
        img_file = request.files["image"]

        uid = str(uuid.uuid4())
        ext = os.path.splitext(img_file.filename)[1]

        in_path = os.path.join(UPLOAD, uid + ext)
        img_file.save(in_path)

        # ================= YOLO =================
        t0 = time.time()
        y = yolo.predict(in_path, conf=conf, verbose=False)[0]
        y_time = round((time.time() - t0) * 1000, 2)

        y_img = cv2.cvtColor(y.plot(), cv2.COLOR_BGR2RGB)
        y_name = f"{uid}_yolo.jpg"

        to_pil_image(
            torch.from_numpy(y_img).permute(2, 0, 1)
        ).save(os.path.join(RESULT, y_name))

        # ================= Faster R-CNN =================
        img = read_image(in_path)
        if img.shape[0] == 4:
            img = img[:3]

        img_in = (img.float() / 255).to(device)

        t1 = time.time()
        out = faster([img_in])[0]
        f_time = round((time.time() - t1) * 1000, 2)

        keep = out["scores"] >= conf
        boxes = out["boxes"][keep]
        scores = out["scores"][keep]
        labels = out["labels"][keep]

        # ===== YOLO-style labels: "Car 0.92" =====
        texts = [
            f"{CLASS_NAMES.get(int(l), 'Obj')} {s:.2f}"
            for l, s in zip(labels, scores)
        ]

        f_img = draw_bounding_boxes(
            img,
            boxes,
            labels=texts,
            width=3,
            colors="green"
        )

        f_name = f"{uid}_faster.jpg"
        to_pil_image(f_img).save(os.path.join(RESULT, f_name))

        # ================= Data to Template =================
        data = {
            "yolo_img": f"results/{y_name}",
            "faster_img": f"results/{f_name}",
            "yolo_time": y_time,
            "faster_time": f_time,
            "yolo_boxes": len(y.boxes) if y.boxes is not None else 0,
            "faster_boxes": len(boxes)
        }

    return render_template("index.html", **data)

# ================= Run =================
if __name__ == "__main__":
    app.run(debug=True)
