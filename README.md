# BDD100K ADAS Perception Pipeline

An end-to-end computer vision pipeline for **data analysis, model training, evaluation, visualization, and failure analysis** on the **BDD100K dataset** for Advanced Driver Assistance Systems (ADAS).

This project investigates the performance of object detection models on real-world driving scenarios and provides insights into model strengths, weaknesses, and opportunities for improvement through comprehensive quantitative and qualitative analysis.

---

## Project Objectives

The project consists of three major stages:

### 1. Dataset Analysis

Perform exploratory data analysis on the BDD100K object detection dataset.

Key objectives:

- Analyze class distributions
- Identify dataset imbalance
- Study object size distributions
- Analyze weather, scene, and time-of-day attributes
- Investigate train vs validation split consistency
- Discover challenging samples and edge cases

A detailed discussion of data analysis is provided in `ANALYSIS.md`.

---

### 2. Model Training

Train and evaluate an object detection model on the BDD100K dataset.

Selected Model:

- YOLO11m

Reasons for selection:

- Strong balance between accuracy and inference speed
- Robust performance on diverse object categories
- Suitable for real-time ADAS applications
- Easy deployment to TensorRT and embedded platforms

A detailed discussion of model selection and architecture is provided in `MODEL_SELECTION.md`.

---

### 3. Evaluation and Visualization

Evaluate model performance using:

- Precision
- Recall
- F1 Score
- Per-class metrics
- Confusion matrix
- Weather-based analysis
- Scene-based analysis
- Time-of-day analysis

Perform qualitative analysis through:

- Ground-truth vs prediction visualization
- Best-case examples
- Failure-case examples
- Failure pattern clustering

A detailed discussion of model evaluation is provided in `EVALUATION.md`.
---




# Dataset

Dataset Used:

**BDD100K (Berkeley DeepDrive)**

Detection Classes:

1. Person
2. Rider
3. Car
4. Truck
5. Bus
6. Train
7. Bike
8. Motor
9. Traffic Light
10. Traffic Sign

BDD100K contains diverse driving conditions including:

- Day
- Night
- Rain
- Snow
- Fog
- Urban streets
- Highways
- Residential roads

making it highly suitable for ADAS perception research.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/<your_username>/bdd100k-adas-perception-pipeline.git

cd bdd100k-adas-perception-pipeline
```

## Create Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

```

Launch dashboard:

```bash
streamlit run data_analysis/app.py
```

Generated outputs include:

- Class distribution
- Object size analysis
- Weather analysis
- Scene analysis
- Time-of-day analysis
- Train vs validation comparison
- Hard example mining

---

# Model Training

Train YOLO11m:

```bash
python model_training/train.py
```

Training outputs:

- Model checkpoints
- Training logs
- Validation metrics
- Loss curves

---

# Model Evaluation

Run evaluation:

```bash
python model_evaluation/evaluate.py
```

Generated outputs:

```text
evaluation/
├── summary.json
├── per_class_metrics.csv
├── confusion_matrix.csv
├── confusion_matrix.png
├── image_level_metrics.csv
├── qualitative/
├── best_cases/
├── failure_cases/
└── attribute_analysis/
```

---

# Evaluation Results

## Overall Performance

| Metric | Score |
|----------|----------|
| Micro Precision | 0.7921 |
| Micro Recall | 0.3400 |
| Micro F1 | 0.4758 |
| Macro Precision | 0.5504 |
| Macro Recall | 0.2680 |
| Macro F1 | 0.3192 |

### Key Findings

- High precision indicates reliable detections.
- Low recall suggests many objects remain undetected.
- Performance varies significantly across object categories.

---

## Best Performing Classes

| Class | F1 Score |
|---------|---------|
| Car | 0.62 |
| Person | 0.54 |
| Bus | 0.47 |

---

## Challenging Classes

| Class | F1 Score |
|---------|---------|
| Rider | 0.00 |
| Traffic Sign | 0.01 |
| Train | 0.03 |

Challenges include:

- Small object sizes
- Class imbalance
- Missing rider representation in COCO pretraining

---

# Failure Analysis

The following recurring failure patterns were identified:

### Small Objects

Affected categories:

- Traffic Signs
- Traffic Lights

### Rider Detection

The COCO-pretrained model lacks a dedicated rider class.

### Night-Time Conditions

Performance drops significantly in low-light environments.

### Long-Range Objects

Objects far from the ego vehicle are frequently missed.

### Crowded Urban Scenes

Occlusion and overlapping objects increase detection difficulty.

---

# Recommendations

### Data Improvements

- Increase traffic sign samples
- Improve rider representation
- Address class imbalance
- Add more night-time samples

### Model Improvements

- Fine-tune on BDD100K
- Increase input resolution
- Use multi-scale feature learning
- Optimize confidence thresholds

---
