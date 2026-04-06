# Natural Disaster Detection using Deep Learning

## Project Overview

This project focuses on building a **deep learning-based image classification system** to detect different types of natural disasters from images.

The system uses **Convolutional Neural Networks (CNNs)** and **transfer learning (ResNet models)** to classify disaster images into categories such as:

*  Flood
*  Fire
*  Earthquake
*  Landslide
*  Smoke / Normal

Additionally, the project includes **model comparison, evaluation metrics, and interpretability techniques (Grad-CAM)**.

---

## Objectives

* Automatically classify natural disaster images
* Compare performance of different deep learning models
* Visualize model decisions using Grad-CAM
* Improve accuracy and reliability for real-world applications

---

## Tech Stack

* Python
* PyTorch
* OpenCV
* NumPy
* Matplotlib

---

## Project Structure

```
Natural_Disaster/
│── train.py                     # Model training
│── test.py                      # Model evaluation
│── split.py                     # Dataset splitting
│── requirements.txt             # Dependencies
│
├── disaster_dataset/            # Raw dataset
├── final_dataset/               # Processed dataset
│
├── models/
│   ├── damage_classifier_resnet18_scratch.pth
│   ├── damage_classifier_resnet50.pth
│
├── results/
│   ├── cm_resnet18_scratch.png
│   ├── cm_resnet50.png
│   ├── Confusion_Matrix_Test.png
│   ├── gradcam_resnet50.png
│   ├── GradCAM_Test.png
│   ├── model_comparison.png
│
└── report/
    └── Damage_Test_Report.pdf
```

---

## Installation & Setup

### 1️. Clone the repository

```
git clone https://github.com/HarshaMittapally/Natural_Disaster_Detection.git
cd Natural_Disaster_Detection
```

### 2️. Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3️. Install dependencies

```
pip install -r requirements.txt
```

---

## Model Training

Run:

```
python train.py
```

This will:

* Train ResNet models
* Save trained weights (`.pth` files)
* Generate training logs

---

## Testing & Evaluation

Run:

```
python test.py
```

 This will:

* Evaluate model performance
* Generate confusion matrix
* Output accuracy metrics

---

## Results & Visualizations

### Confusion Matrix

Shows classification performance across all classes.

### Model Comparison

Compares ResNet18 vs ResNet50 performance.

### Grad-CAM Visualization

Highlights important regions used by the model for prediction.

---

## Models Used

| Model    | Description                            |
| -------- | -------------------------------------- |
| ResNet18 | Lightweight model trained from scratch |
| ResNet50 | Deep model with higher accuracy        |

---

## Key Features

* Deep learning-based image classification
* Transfer learning using ResNet
* Model performance comparison
* Grad-CAM explainability
* Structured dataset pipeline

---

## Applications

* Disaster monitoring systems
* Emergency response planning
* Satellite image analysis
* Smart surveillance systems

---

## Limitations

* Requires large dataset for better accuracy
* Sensitive to image quality
* May misclassify similar-looking disasters

---

## Future Improvements

* Add more disaster categories
* Deploy as web application
* Integrate real-time detection
* Improve dataset diversity
* Use advanced models (EfficientNet, Vision Transformers)

---

## Learning Outcomes

* Deep learning model training using PyTorch
* Dataset preprocessing and handling
* Model evaluation and comparison
* Explainable AI (Grad-CAM)
* GitHub project management

---
