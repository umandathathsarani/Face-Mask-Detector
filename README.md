# 😷 Real-Time Face Mask Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

### Real-Time Face Mask Detection Using Deep Learning & Computer Vision

An end-to-end Machine Learning project that detects human faces and classifies whether they are wearing a face mask in real-time using OpenCV and a fine-tuned MobileNetV2 model.

</div>

---

# 📌 Table of Contents

* Overview
* Features
* Project Architecture
* Dataset
* Exploratory Data Analysis
* Model Development
* Model Comparison
* Results
* Web Application
* Installation
* Usage
* Project Structure
* Future Improvements
* Tech Stack
* License
* Author

---

# 🚀 Overview

This project was developed to explore the complete Machine Learning lifecycle, from data analysis and visualization to model training, optimization, and deployment.

The system uses:

* OpenCV Haar Cascades for face detection
* Transfer Learning with MobileNetV2
* Real-time webcam inference
* Streamlit for deployment

The final model is lightweight, accurate, and suitable for real-time applications on consumer hardware.

---

# ✨ Features

## 📊 Exploratory Data Analysis

* Class distribution analysis
* Image dimension analysis
* Principal Component Analysis (PCA)
* t-SNE visualization
* Eigenface generation

## 🧠 Deep Learning

* Transfer Learning
* MobileNetV2 fine-tuning
* Data augmentation
* Early stopping
* Learning-rate scheduling
* Model checkpointing

## 🎥 Real-Time Detection

* Webcam integration
* Face detection
* Live prediction
* Confidence score display
* Bounding box visualization

## 🌐 Deployment

* Streamlit web application
* Fast inference pipeline
* Easy local setup

---

# 🏗️ Project Architecture

```text
Input Image / Webcam
          │
          ▼
Face Detection (OpenCV Haar Cascade)
          │
          ▼
Face Cropping & Preprocessing
          │
          ▼
MobileNetV2 Classifier
          │
          ▼
Mask / No Mask Prediction
          │
          ▼
Streamlit Interface
```

---

# 📂 Dataset

The dataset consists of two classes:

| Class        | Description                             |
| ------------ | --------------------------------------- |
| With Mask    | Images of people wearing face masks     |
| Without Mask | Images of people not wearing face masks |

### Preprocessing

* Image resizing
* Normalization
* Data augmentation
* Train / Validation / Test split

---

# 📈 Exploratory Data Analysis

Before model training, multiple techniques were used to understand the dataset.

## Principal Component Analysis (PCA)

PCA was used to:

* Reduce dimensionality
* Understand variance distribution
* Visualize dominant image patterns

## t-SNE Visualization

t-SNE helped visualize:

* Cluster formation
* Class separability
* Feature relationships

## Eigenfaces

Eigenfaces were generated to:

* Understand dominant facial structures
* Analyze dataset variability
* Visualize learned facial representations

---

# 🧠 Model Development

## Base Architecture

### MobileNetV2

MobileNetV2 was selected because it provides:

* Fast inference
* Lightweight architecture
* Low memory consumption
* Strong performance on image classification tasks

### Training Pipeline

```text
Pretrained MobileNetV2
          │
          ▼
Replace Classification Head
          │
          ▼
Fine-Tune Network
          │
          ▼
Validation
          │
          ▼
Save Best Model
```

---

# 🔍 Model Comparison

| Model           | Parameters | Accuracy* | Inference Speed | Model Size |
| --------------- | ---------- | --------- | --------------- | ---------- |
| ResNet18        | ~11.7M     | High      | Medium          | Large      |
| MobileNetV2 ⭐   | ~3.5M      | 98%+      | Fast            | Small      |
| EfficientNet-B0 | ~5.3M      | High      | Medium          | Medium     |

### Why MobileNetV2?

MobileNetV2 was chosen because it provides:

* Faster inference
* Lower memory usage
* Smaller deployment size
* Better real-time performance

### Summary

| Criterion              | Best Model    |
| ---------------------- | ------------- |
| Fastest Inference      | MobileNetV2   |
| Smallest Model         | MobileNetV2   |
| Best Deployment Choice | MobileNetV2 ⭐ |

*Accuracy values may vary depending on dataset, preprocessing, and training configuration.

---

# 📊 Results

## Performance Metrics

| Metric              | Value |
| ------------------- | ----- |
| Validation Accuracy | 98%+  |
| Precision           | High  |
| Recall              | High  |
| F1 Score            | High  |

### Training Curves

```markdown
![Accuracy Curve](assets/accuracy_curve.png)

![Loss Curve](assets/loss_curve.png)
```

### Confusion Matrix

```markdown
![Confusion Matrix](assets/confusion_matrix.png)
```

---

# 🌐 Web Application

The project includes a Streamlit-based interface.

## Features

* Upload image prediction
* Real-time webcam detection
* Confidence score display
* Interactive interface

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/umandathathsarani/Face-Mask-Detector.git

cd Face-Mask-Detector
```

## Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

## Run Streamlit Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Run Real-Time Detection

```bash
python realtime_detection.py
```

---

# 📂 Project Structure

```text
Face-Mask-Detector/
│
├── assets/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   ├── app_demo.png
│   └── realtime_demo.gif
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Training.ipynb
│   └── 04_RealTimeDetection.ipynb
│
├── saved_models/
│   └── mobilenetv2_mask_detector.pth
│
├── app.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 🔮 Future Improvements

* YOLO-based face detection
* Multi-face tracking
* ONNX export
* TensorRT optimization
* Docker deployment
* AWS deployment
* Mobile deployment
* Raspberry Pi deployment

---

# 🛠️ Tech Stack

## Machine Learning

* PyTorch
* Torchvision
* Scikit-Learn

## Computer Vision

* OpenCV

## Data Analysis

* NumPy
* Pandas
* Matplotlib
* Seaborn

## Deployment

* Streamlit

## Development

* Jupyter Notebook
* Git
* GitHub
* VS Code

---

# 📜 License

This project is licensed under the MIT License.

See the LICENSE file for details.

---

# 👩‍💻 Author

**Mummullage Binuri Umanda Thathsarani**

Bachelor of Science (Hons) in Information Technology
Specialization in Artificial Intelligence

GitHub:
https://github.com/umandathathsarani

---

# ⭐ Support

If you found this project useful:

* Star the repository
* Fork the repository
* Share feedback
* Contribute improvements

Thank you for visiting the project!
