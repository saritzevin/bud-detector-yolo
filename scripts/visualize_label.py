from ultralytics import YOLO
from pathlib import Path
import random
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scripts.config import Config


def main(cfg):
    vis_cfg = cfg.section("visualize")
    infer_cfg = cfg.section("inference")

    num_samples = vis_cfg.get("num_samples", 4)

    # Load trained model (best.pt)
    model = YOLO(cfg["weights_path"])

    # Dataset images
    images_dir = Path(cfg["path"], "images")
    imgs = list(images_dir.glob("*"))
    samples = random.sample(imgs, min(num_samples, len(imgs)))

    for img_path in samples:
        # Derive GT label path
        label_path = str(img_path).replace("images", "labels")
        label_path = label_path.replace(".jpg", ".txt")

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        vis = img.copy()

        # =========================
        # Draw GT (Ground Truth)
        # =========================
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
        else:
            print(f"[WARNING] Missing GT: {label_path}")
            lines = []

        for line in lines:
            vals = list(map(float, line.strip().split()))
            if len(vals) != 5:
                continue

            _, xc, yc, bw, bh = vals

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            # Red = GT
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # =========================
        # Draw Predictions
        # =========================
        results = model.predict(
            source=str(img_path),
            conf=infer_cfg.get("conf", 0.1),
            iou=infer_cfg.get("iou", 0.5),
            verbose=False
        )

        boxes = results[0].boxes

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Green = Prediction
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # =========================
        # Show result
        # =========================
        plt.figure(figsize=(10, 8))
        plt.imshow(vis)
        plt.title(img_path.name)
        plt.axis("off")

        red_patch = mpatches.Patch(color='red', label='Ground Truth')
        green_patch = mpatches.Patch(color='green', label='Prediction')

        plt.legend(handles=[red_patch, green_patch], loc='upper right')

        plt.show()

if __name__ == '__main__':
    cfg = Config("data/config.yaml")
    main(cfg)