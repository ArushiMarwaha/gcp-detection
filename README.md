# Aerial GCP Detection

## Overview

This project implements a computer vision pipeline to automatically detect Ground Control Points (GCPs) in aerial drone imagery.

Given an aerial image containing a GCP marker, the system predicts:

1. The pixel coordinates of the **center of the marker**
2. The **shape of the marker**, which can be one of:
   - Cross
   - Square
   - L-Shaped

Automating this process is important for aerial surveying and photogrammetry workflows, where GCP markers are used to align drone imagery with precise geographic coordinates.

---

## Approach

The solution uses a **Convolutional Neural Network (CNN)** with a pretrained **ResNet18 backbone** to extract visual features from aerial images.

Two prediction heads are added on top of the feature extractor:

1. **Coordinate Head**
   - Outputs two values `(x, y)`
   - Predicts the pixel location of the marker center

2. **Shape Classification Head**
   - Outputs three class probabilities
   - Predicts the marker type (`Cross`, `Square`, `L-Shaped`)

This design allows the model to perform **multi-task learning**, where localization and classification are learned simultaneously.

---

## Data Processing

Images are resized before being passed to the neural network to reduce computational cost.

During preprocessing:

- Images are resized to **256 × 256**
- Marker coordinates are **normalized relative to the image width and height**
- Images are converted to tensors and scaled to `[0,1]`

The dataset follows a nested directory structure:
project_name / survey_name / gcp_id / image.JPG


Relative paths are preserved during training and inference to maintain compatibility with the dataset format.

---

## Training Strategy

The model is trained using a combination of two loss functions.

### Coordinate Regression Loss

Mean Squared Error (MSE) is used to measure the difference between predicted and ground-truth marker coordinates.
Coordinate Loss = MSE(predicted_xy, true_xy)


### Shape Classification Loss

Cross Entropy Loss is used for marker shape classification.
Shape Loss = CrossEntropy(predicted_shape, true_shape)


### Total Loss
Total Loss = Coordinate Loss + Shape Loss


The model is trained using the **Adam optimizer**.

---

## Dataset Handling

During dataset exploration, a few practical issues were identified:

- Some annotation entries referenced images that were **not present locally**
- A small number of annotations were **missing the `verified_shape` field**

To ensure robust training, these entries were **filtered during dataset preparation**, and only valid labeled samples were used.

This reflects common real-world scenarios where production datasets may contain inconsistencies or incomplete annotations.

---

## Running the Project

### 1. Install Dependencies

Install all required Python packages:
pip install -r requirements.txt


---

### 2. Train the Model

Run the training script:
python src/train.py


This script:

- loads the dataset
- filters valid samples
- trains the neural network
- saves the trained model

Model weights are stored in:
outputs/gcp_model.pth


---

### 3. Run Inference on the Test Dataset

Run the inference script:

python src/inference.py



This script:

- loads the trained model
- processes all images in the test dataset
- generates predictions for marker location and shape

The final predictions are saved as:

outputs/predictions.json

---

## Output Format

The generated `predictions.json` follows the same format as the training annotations.

Example:
{
  "231129_CTD/231129_CTD_GDA94/230225gcp11/DJI_20231129114405_0532.JPG": {
    "mark": {
      "x": 2647.96,
      "y": 1578.10
    },
    "verified_shape": "Cross"
  },
  "231129_CTD/231129_CTD_GDA94/GCP9/DJI_20231129125237_0191.JPG": {
    "mark": {
      "x": 1205.80,
      "y": 3021.90
    },
    "verified_shape": "Cross"
  }
}


Each entry contains:

- the relative image path
- the predicted `(x, y)` coordinates of the marker center
- the predicted marker shape

---

## Project Structure

gcp-detection
│
├── notebooks
│ └── eda.ipynb
│
├── src
│ ├── dataset.py
│ ├── model.py
│ ├── train.py
│ └── inference.py
│
├── outputs
│ ├── gcp_model.pth
│ └── predictions.json
│
├── requirements.txt
└── README.md


---

## Summary

This project demonstrates a practical computer vision pipeline for detecting GCP markers in aerial imagery. The solution combines:

- dataset exploration and preprocessing
- a CNN-based multi-task learning model
- robust dataset filtering to handle incomplete annotations
- a reproducible training and inference pipeline
- generation of predictions in the required JSON format

The final system is simple, reproducible, and suitable for integration into larger aerial surveying workflows.





