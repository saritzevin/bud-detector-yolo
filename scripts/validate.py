from pathlib import Path
import json

from ultralytics import YOLO
from scripts.config import Config


def main(cfg):
    val_cfg = cfg.section("validation")  # validation config section

    weights = cfg["weights_path"]        # path to best trained weights
    data = val_cfg["data"]               # dataset YAML (bud.yaml)

    # sanity check: weights exist
    if not Path(weights).exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    # sanity check: dataset config exists
    if not Path(data).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")

    # load trained model
    model = YOLO(weights)

    # run validation (compute metrics on val/test split)
    metrics = model.val(
        data=data,
        split=val_cfg["split"],   # val / test
        imgsz=val_cfg["imgsz"],   # input resolution
        conf=val_cfg["conf"],     # low threshold for mAP calc
        iou=val_cfg["iou"],       # NMS IoU threshold
        plots=val_cfg["plots"],   # save PR curves etc.
    )

    # extract main metrics
    results = {
        "split": val_cfg["split"],
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }

    # where to save metrics JSON
    out_path = Path(val_cfg.get("output_metrics", "reports/metrics.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # save metrics to file
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # print results for quick inspection
    print(json.dumps(results, indent=2))
    print(f"Saved metrics to: {out_path}")


if __name__ == '__main__':
    cfg = Config("data/config.yaml")
    main(cfg)