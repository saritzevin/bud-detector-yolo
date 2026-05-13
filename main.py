from scripts.config import Config

from scripts import train_model, validate, infer_dir, visualize_label, prepare_dataset


# Full pipeline: data -> train -> validate -> infer -> visualize
def run(cfg_path):
    # Load configuration
    cfg = Config(cfg_path)

    # Prepare dataset (split + convert labels) – usually run once
    # prepare_dataset.main(cfg)

    # Train YOLO model and save best weights
    train_model.main(cfg)

    # Evaluate model on validation set (metrics like mAP, recall)
    validate.main(cfg)

    # Run inference on test images and save predictions
    infer_dir.main(cfg)

    # Visualize random samples with predictions
    visualize_label.main(cfg)


if __name__ == '__main__':
    run(cfg_path="data/config.yaml")


# todo - in readme - add results!!!
# todo - in DESIGN_AND_REASONING - add results!!!
