# BDD100K Dataset Analytics: Object Detection for ADAS

## Overview

This repository contains an end-to-end Exploratory Data Analysis (EDA) pipeline and interactive dashboard for the BDD100K dataset (100K subset), specifically focused on the 10-class 2D object detection task.

The project evaluates class distributions, geometric priors, object scales, environmental context, and identifies edge cases/anomalies critical for training robust Advanced Driver Assistance Systems (ADAS) perception models.

### Key Components

1. **Interactive Dashboard**
   - Streamlit application for dynamic data visualization and qualitative edge-case exploration.

2. **Analysis Report**
   - Comprehensive engineering document (`ANALYSIS.md`) containing actionable insights derived from the dataset.

3. **Data Parser**
   - Optimized parsing logic to process BDD100K JSON labels and extract dataset statistics.




---

# Prerequisites & Data Setup

To keep the container lightweight and strictly self-contained regarding dependencies, the raw BDD100K data (images and labels) must be mounted as a volume at runtime.

### Requirements

- Docker installed on your system
- BDD100K Dataset (100K Images + Labels)

### Download Dataset

Download the following components from the official BDD100K release:

- Images (100K subset) (~5.3 GB)
- Labels (~107 MB)



---

## Important Note on Image Visualization

> The statistical dashboards and metric charts are generated using the pre-computed JSON reports.
>
> However, the **"Interesting & Hard Samples"** visualizer relies on the original `.jpg` image files.
>
> Visualizing edge cases and bounding-box overlays is only possible if the dataset directory is correctly mounted to the Docker container during execution.
>
> If images are unavailable, the dashboard will gracefully display a warning message without crashing.

---

# Running the Project

The project is fully containerized. No local Python environment or package installation is required.

---

## Step 1: Build the Docker Image
Ensure Docker Desktop is running on your system before proceeding.
Navigate to the repository root containing docker file and execute:


```bash
docker build -t bdd100k-analytics .
```

---

## Step 2: Run the Container

Mount the dataset directory to the container and expose the Streamlit port.

Replace:

```text
/path/to/local_bdd_data
```

with the absolute path of your dataset folder.

### Linux / macOS

```bash
docker run -p 8501:8501 \
  -v /path/to/local_bdd_data:/app/data \
  bdd100k-analytics
```

### Windows (PowerShell)

```powershell
docker run -p 8501:8501 `
  -v E:\BDD100K\local_bdd_data:/app/data `
  bdd100k-analytics
```

### Port Configuration

If port **8501** is already occupied:

```bash
docker run -p 8502:8501 \
  -v /path/to/local_bdd_data:/app/data \
  bdd100k-analytics
```

The dashboard will then be available at:

```text
http://localhost:8502
```

---

## Step 3: Access the Dashboard

### Interactive Dashboard

Open your browser and navigate to:

```text
http://localhost:8501
```

### Engineering Analysis Report

The complete written analysis is available in:

```text
ANALYSIS.md
```

---

# Dashboard Features

### Dataset Overview

- Total images and annotations
- Train vs Validation statistics
- Class frequency distribution

### Object Geometry Analysis

- Bounding-box area distribution
- Aspect-ratio analysis
- Object scale categorization
- Small, medium, and large object statistics

### Environmental Context Analysis

- Weather conditions
- Scene attributes
- Time-of-day distribution

### Train vs Validation Comparison

- Dataset split consistency
- Class distribution comparison
- Domain shift identification

### Interesting & Hard Samples

- Largest objects
- Smallest objects
- Crowded scenes
- Dense traffic scenarios
- Rare-class examples
- Edge-case visualization

### Engineering Insights

- Dataset bias identification
- Class imbalance assessment
- Recommendations for model training
- ADAS-specific deployment considerations

---

# Generated Reports

The parser automatically generates statistical reports:

```text
analysis_reports/
├── train_report.json
└── val_report.json
```

These reports contain:

- Dataset statistics
- Class distributions
- Geometric properties
- Environmental metadata
- Hard-sample summaries

The dashboard reads directly from these reports for fast loading and visualization.

