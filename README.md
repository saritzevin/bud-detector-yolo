# Bud Detector – YOLO Pipeline

## Overview

This project implements an end-to-end pipeline for detecting **buds** in images using a YOLO-based model.

The pipeline includes:
- Dataset preparation (polygon → bounding boxes)
- Model training (fine-tuning pretrained YOLO)
- Validation (precision, recall, mAP)
- Inference
- Visualization (ground truth vs predictions)

---

## Configuration

All parameters are controlled via a single YAML file:

config.yaml

This includes:
- Training settings
- Augmentations
- Validation thresholds
- Inference parameters

---

## Run Pipeline

python main.py

---

## Dataset Preparation

python scripts/prepare_dataset.py

- Converts polygon annotations to YOLO bounding boxes
- Splits dataset into train/val/test

---

## Training

python scripts/train_model.py

- Uses pretrained YOLO weights
- Trains on custom dataset
- Saves best model automatically

---

## Validation

python scripts/validate.py

Outputs:
- Precision
- Recall
- mAP

---

## Inference

python scripts/infer_dir.py

Generates predictions on test images.

---

## Visualization

python scripts/visualize_label.py

Displays:
- Ground Truth
- Predictions

---

## Results (Summary)

- Precision: 0.67  
- Recall: 0.45 

The model is conservative: predictions are relatively accurate, but many objects are missed.

---

## Notes

- Validation uses low confidence (conf=0.01)
- Inference uses higher thresholds (conf=0.1)

---

## Conclusion

This project demonstrates a full detection pipeline with a focus on reproducibility, modularity, and evaluation.