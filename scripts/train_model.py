from ultralytics import YOLO
import random
import numpy as np
import torch
from scripts.config import Config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(cfg):
    train_cfg = cfg.section("training")
    params = train_cfg["params"]

    # core training arguments
    train_args = {
        "data": train_cfg["data"],
        "imgsz": train_cfg["imgsz"],
        "epochs": train_cfg["epochs"],
        "batch": train_cfg["batch"],
        "project": train_cfg["project"],
        "name": train_cfg["name"],
        "seed": train_cfg.get("seed", 42),
    }

    train_args.update(params)

    model_name = train_cfg["model"]

    # load model (with pretrained weights)
    model = YOLO(model_name)

    # run training
    model.train(**train_args)

    # save best checkpoint path into config (for later inference/validation)
    cfg.cfg['weights_path'] = str(model.trainer.best)

if __name__ == '__main__':
    cfg = Config("data/config.yaml")
    main(cfg)