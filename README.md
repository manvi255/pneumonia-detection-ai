# Pneumonia Detection using Residual CNN + Grad-CAM

> Deep learning system for automated pneumonia detection from chest X-ray images using a custom Residual CNN architecture, threshold tuning, and Grad-CAM explainability.

---

## Overview

This project explores the use of deep convolutional neural networks for detecting **Pneumonia** from chest X-ray images.

Unlike standard beginner implementations based entirely on transfer learning, this system uses a **custom Residual CNN architecture** built completely from scratch in **PyTorch**.

The project focuses not only on classification accuracy, but also on:

* Explainable AI
* Threshold Optimization
* Residual Learning
* Class Imbalance Handling
* Medical AI Deployment

The final system is deployed as an interactive **Streamlit web application** with integrated **Grad-CAM visualization**.

---

# Model Architecture

The network is composed of:

```text
Input (224×224×1)
    ↓
Conv(32)
    ↓
BatchNorm + ReLU + MaxPool
    ↓
Residual Block (64)
    ↓
Residual Block (128)
    ↓
Global Average Pooling
    ↓
FC(256)
    ↓
Dropout(0.5)
    ↓
FC(2)
    ↓
Softmax
```

## Key Architectural Features

* Residual learning for stable gradient flow
* Batch normalization for faster convergence
* Global Average Pooling to reduce overfitting
* Dropout regularization
* Threshold tuning for improved class balance

---

# Dataset

Dataset used:

* Chest X-ray Pneumonia Dataset
* Source: Kaggle

Classes:

* NORMAL
* PNEUMONIA

Challenges faced:

* severe class imbalance
* high pneumonia sensitivity
* false positive predictions on normal lungs

---

# Data Augmentation

Training augmentations:

```python
RandomHorizontalFlip()
RandomRotation(12)
RandomAffine()
Resize((224,224))
Normalize([0.5],[0.5])
```

These augmentations improved model generalization and robustness.

---

# Training

## Loss Function

```python
CrossEntropyLoss
```

## Optimizer

```python
Adam
```

## Regularization

```python
Dropout(0.5)
```

Additional optimization techniques:

* threshold tuning
* learning rate scheduling
* class balancing

---

# Threshold Tuning

Instead of using the default threshold of **0.5**, multiple thresholds were experimentally tested:

```text
0.70
0.75
0.79
0.82
0.92
0.94
```

Final deployment threshold:

```text
0.80
```

This improved:

* normal class accuracy
* prediction balance
* false positive reduction

while maintaining strong pneumonia detection performance.

---

# Final Results

| Metric   | Score  |
| -------- | ------ |
| Accuracy | 89.74% |
| F1 Score | 0.922  |

## Confusion Matrix

```text
[[181  53]
 [ 11 379]]
```

---

# Explainable AI (Grad-CAM)

Grad-CAM visualization was integrated to make predictions interpretable.

The generated heatmaps highlight the lung regions that contributed most strongly to the prediction.

This transforms the model from a black-box classifier into a more transparent and explainable medical AI system.

---

# Streamlit Deployment

Features of deployed application:

* Upload chest X-ray images
* Real-time prediction
* Confidence scores
* Grad-CAM heatmaps
* Professional medical dashboard UI

---

# Tech Stack

```text
PyTorch
Torchvision
Streamlit
OpenCV
NumPy
Pillow
```

---

# Run Locally

Clone repository:

```bash
git clone <your_repo_url>
cd pneumonia-detection-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
python -m streamlit run app.py
```

---

# Future Improvements

Potential future improvements:

* Ensemble learning
* DenseNet / EfficientNet architectures
* Hard Negative Mining
* Multi-class lung disease detection
* Cloud deployment
* Clinical report generation

---

# Project Goals

This project was built to explore:

* medical image analysis
* deep learning system design
* explainable AI
* deployment of real-world AI applications

while deve
