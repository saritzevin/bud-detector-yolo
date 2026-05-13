from ultralytics import YOLO
from pathlib import Path

from scripts.config import Config


def main(cfg):
    infer_cfg = cfg.section("inference")  # inference config section

    weights = cfg["weights_path"]         # path to best trained weights
    source = infer_cfg["source"]          # input images directory
    output = infer_cfg["output"]          # output folder

    # load trained model
    model = YOLO(weights)

    # sanity check: source exists
    Path(source).exists() or (_ for _ in ()).throw(
        FileNotFoundError(f"Source not found: {source}")
    )

    # run inference and save predictions
    model.predict(
        source=source,
        imgsz=infer_cfg["imgsz"],   # input resolution
        conf=infer_cfg["conf"],     # confidence threshold (controls FP/recall)
        iou=infer_cfg["iou"],       # NMS IoU threshold

        save=True,                  # save images with boxes
        save_txt=infer_cfg["save_txt"],   # save predictions as txt
        save_conf=infer_cfg["save_conf"], # save confidence scores

        project=output,             # root output folder
        name="results",             # subfolder name
        exist_ok=True,              # overwrite if exists
        show_labels=False,
    )

    print(f"Predictions saved to: {output}/results")


if __name__ == '__main__':
    cfg = Config("data/config.yaml")
    main(cfg)