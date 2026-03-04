# Aerial GCP Detection

This repository implements a computer vision pipeline for the automated detection and classification of Ground Control Points (GCPs) in aerial drone imagery.

Automating GCP detection is a critical component of photogrammetry and aerial surveying workflows, enabling the precise alignment of drone imagery with geographic coordinate systems.

---

## Project Overview

The system processes aerial imagery to predict two primary attributes for each GCP marker:

1. **Localization:** Pixel coordinates $(x, y)$ of the marker's center.
2. **Classification:** The geometric shape of the marker, categorized as:
* **Cross**
* **Square**
* **L-Shaped**



---

## Technical Approach

The solution utilizes a **Convolutional Neural Network (CNN)** built upon a pretrained **ResNet18** backbone for feature extraction. The architecture employs a **Multi-Task Learning** design by branching into two specialized prediction heads:

### 1. Coordinate Regression Head

* **Output:** Continuous values for $(x, y)$.
* **Function:** Regresses the precise pixel location of the marker center.

### 2. Shape Classification Head

* **Output:** Class probabilities for the three marker types.
* **Function:** Categorizes the visual geometry of the detected marker.

---

## Data Processing and Engineering

### Preprocessing Pipeline

To optimize computational efficiency and model convergence, images undergo the following transformations:

* **Resizing:** All input images are scaled to $256 \times 256$ pixels.
* **Normalization:** Coordinates are normalized relative to image dimensions; pixel values are scaled to the $[0, 1]$ range.
* **Data Integrity:** The pipeline automatically filters entries with missing local image files or incomplete metadata (e.g., missing `verified_shape` fields) to ensure a high-quality training set.

### Directory Structure

The dataset follows a hierarchical format that is preserved during inference to maintain compatibility with existing photogrammetry software:
`project_name / survey_name / gcp_id / image.JPG`

---

## Training Methodology

The model is optimized using the **Adam optimizer** and a composite loss function to balance localization accuracy and classification precision.

### Loss Functions

* **Coordinate Loss:** Calculated using Mean Squared Error (MSE).

$$L_{coord} = \text{MSE}(\hat{y}_{xy}, y_{xy})$$


* **Shape Loss:** Calculated using Cross-Entropy Loss.

$$L_{shape} = \text{CrossEntropy}(\hat{y}_{class}, y_{class})$$


* **Total Objective:** 
$$\text{Total Loss} = L_{coord} + L_{shape}$$



---

## Implementation Guide

### 1. Environment Setup

Install the necessary dependencies using pip:

```bash
pip install -r requirements.txt

```

### 2. Model Training

Execute the training script to filter the dataset, train the multi-task network, and serialize the weights:

```bash
python src/train.py

```

* **Output:** `outputs/gcp_model.pth`

### 3. Inference

Run the inference engine on the test dataset to generate predictions:

```bash
python src/inference.py

```

* **Output:** `outputs/predictions.json`

---

## Output Specifications

The resulting `predictions.json` adheres to the standard annotation format, ensuring seamless integration with downstream surveying tools.

```json
{
  "231129_CTD/231129_CTD_GDA94/230225gcp11/DJI_20231129114405_0532.JPG": {
    "mark": {
      "x": 2647.96,
      "y": 1578.10
    },
    "verified_shape": "Cross"
  }
}

```

---

## Repository Structure

```text
gcp-detection
│
├── notebooks
│   └── eda.ipynb           # Exploratory Data Analysis
│
├── src
│   ├── dataset.py         # Data loading and augmentation
│   ├── model.py           # Model architecture (ResNet18 + MTL Heads)
│   ├── train.py           # Training logic and loss optimization
│   └── inference.py       # Prediction and JSON generation
│
├── outputs
│   ├── gcp_model.pth      # Trained model weights
│   └── predictions.json   # Inference results
│
├── requirements.txt
└── README.md

```
