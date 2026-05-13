# Design & Reasoning Document

## Goal

The goal of this project is to demonstrate a structured approach to problem solving, including analysis, design, and evaluation.

- Understanding of the problem  
- Thoughtful design decisions  
- Awareness of tradeoffs  
- Ability to analyze model behavior  

---

## Problem Analysis

The dataset presents several key challenges:

1. Small objects: Buds occupy very few pixels, making them difficult to detect after downsampling.  
2. Dense scenes: Multiple buds appear close together, leading to suppression during NMS.  
3. Limited dataset (~100 images): High risk of overfitting and poor generalization.  

---

## Annotation Format Decision

The original dataset was provided as segmentation polygons.

### Decision:
Convert polygons to bounding boxes and use YOLO detection.

### Reasoning:

- Detection models are more stable with small datasets  
- Lower computational complexity  
- Faster training and iteration  

### Tradeoff:

- Loss of precise object boundaries  
- Reduced performance for small, irregular shapes  

---

## Data Augmentation Strategy

Due to limited data and small objects, augmentation is critical.

### Augmentations used:

- **Mosaic** increases object density and diversity  
- **Scaling and translation** improve robustness to size and position  
- **Horizontal flip** adds spatial invariance  
- **HSV (color jitter)** improves robustness to lighting conditions  

---

## Why augmentation is important

Augmentations were used to address key challenges:

- Small objects: scaling and mosaic improve visibility  
- Limited data: increases effective dataset size  
- Dense scenes: mosaic improves robustness to clutter  

---

## Tradeoffs

- Strong augmentations may create unrealistic samples  
- Too much augmentation can harm convergence  

---

## Model Choice

YOLO detection model with pretrained weights.

### Why?

- Strong baseline for detection  
- Efficient and well-supported  
- Transfer learning reduces data requirements  

---

## Input Resolution

imgsz = 1024

### Reasoning:

- Small objects require high resolution  
- Prevents loss of spatial information  

### Tradeoff:

- Slower training  

---

## Training Design

- AdamW optimizer  
- Data augmentation enabled  
- Early stopping  

### Reasoning:

- Stable optimization  
- Better generalization  
- Prevent overfitting  

---

## Validation Strategy

conf = 0.01

### Why?

- Ensures most predictions are considered  
- Enables proper precision-recall curve estimation  
- Reduces excessive NMS computation compared to extremely low thresholds  

---

## Inference Strategy

conf = 0.1

### Why?

- Reduce false positives  
- Improve usability  

---

## Results Analysis

- Precision: 0.67  
- Recall: 0.45  
- mAP50: 0.47  
- mAP50_95: 0.23  

## Interpretation

The model demonstrates a balanced performance with moderate precision and recall, indicating that it is able to detect a substantial portion of the buds while maintaining relatively reliable predictions.

The results reflect the challenging nature of the task, particularly due to the small object size and high object density.

The relatively lower mAP50_95 suggests that precise localization remains difficult, especially for very small objects and dense regions.

With additional time, further improvements could be explored, such as the approaches outlined in the Future Improvements section.

Overall, the model provides a reasonable tradeoff between detection coverage and prediction reliability.

---

## Evaluation Metrics

The following metrics were tracked:

- Precision – reliability of predictions  
- Recall – coverage of detected objects  
- mAP – overall detection performance  

Given the problem characteristics, recall was a key focus.

---

## Class Imbalance

The dataset contains a large amount of background compared to the number of bud instances, and the objects are very small.

This was addressed indirectly through:

- Mosaic augmentation (increasing object density)  
- Lower confidence thresholds (improving recall)  
- High input resolution (preserving small objects)  

---

## Overfitting Mitigation

To reduce overfitting given the small dataset:

- Used pretrained weights (transfer learning)  
- Applied data augmentation (mosaic, scaling, HSV)  
- Used weight decay regularization  
- Applied early stopping (patience)  

---

## Future Improvements

- Use segmentation models for small and dense objects  
- Increase dataset size  
- Tune NMS and inference thresholds  
- Apply multiscale training  
- Use pseudo-labeling (PGT) to expand training data  

---

## Key Takeaway

Every design decision in this project was driven by:

- Dataset characteristics  
- Model behavior  
- Practical constraints  

The focus was not only on performance, but on understanding the system and its limitations.