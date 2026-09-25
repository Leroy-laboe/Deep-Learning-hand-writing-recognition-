# Handwriting Author Identification

**A deep-learning application that identifies the writer of a handwriting sample using computer vision, character segmentation, and a custom CNN classifier.**

Built as a practical Deep Learning / Computer Vision project with a **FastAPI inference API** and a lightweight browser interface.

![Accuracy](https://img.shields.io/badge/Recorded_Test_Accuracy-83.57%25-success)
![Writers](https://img.shields.io/badge/Writer_Classes-70-6f42c1)
![Stack](https://img.shields.io/badge/Stack-TensorFlow%20%7C%20OpenCV%20%7C%20FastAPI-blue)

---

## Overview

This project tackles **handwriting authorship identification** rather than OCR.

Given an image containing handwriting, the system processes the sample, extracts handwriting segments, classifies those segments with a convolutional neural network, and combines the predictions to identify which known writer produced the sample.

The trained model currently supports **70 writer classes**.

### Recorded evaluation result

**83.57% test accuracy**

This is the result recorded when evaluating the trained model through the project's Python test pipeline.

---

## How It Works

```text
Handwriting Image
       │
       ▼
Grayscale Conversion
       │
       ▼
Otsu Thresholding
       │
       ▼
Line Segmentation
       │
       ▼
Word Segmentation
       │
       ▼
Character / Segment Extraction
       │
       ▼
Resize + Normalize to 64 × 64
       │
       ▼
CNN Predictions
       │
       ▼
Aggregate Segment Predictions
       │
       ▼
Predicted Writer ID
```

Instead of relying on one crop from the uploaded image, the system analyses multiple handwriting segments and combines their predictions for the final result.

---

## Key Features

### Writer identification
Classifies handwriting samples across **70 known writers**.

### Computer-vision preprocessing
Uses OpenCV for:

- grayscale conversion;
- Otsu thresholding;
- line segmentation;
- word segmentation;
- character / handwriting-segment extraction;
- image resizing and normalization.

### Custom CNN model
The classifier was built with TensorFlow / Keras rather than using an external AI API.

The network uses multiple convolutional blocks with:

- Conv2D layers;
- batch normalization;
- max pooling;
- dropout;
- global average pooling;
- a dense classification head;
- softmax output across the writer classes.

### Data augmentation
Training includes rotation, translation, zoom, contrast changes, Gaussian noise, and additional character-level augmentation to improve robustness.

### Multi-segment prediction
Each detected handwriting segment is classified independently and the individual predictions are combined to determine the final writer.

### Low-confidence rejection
The inference API includes a confidence threshold so very low-confidence inputs can be returned as **Unknown** rather than always forcing a writer prediction.

### Web inference interface
A browser UI allows a user to:

1. upload a handwriting sample;
2. preview the image;
3. send it to the FastAPI backend;
4. view the predicted writer;
5. inspect the displayed prediction confidence.

---

## Model Architecture

```text
Input: 64 × 64 × 3
       │
       ▼
Conv2D 32 + BatchNorm + MaxPool + Dropout
       │
       ▼
Conv2D 64 + BatchNorm + MaxPool + Dropout
       │
       ▼
Conv2D 128 + BatchNorm + MaxPool + Dropout
       │
       ▼
Conv2D 256 + BatchNorm + MaxPool + Dropout
       │
       ▼
Conv2D 256 + BatchNorm + MaxPool + Dropout
       │
       ▼
Global Average Pooling
       │
       ▼
Dense 1024 + BatchNorm + Dropout
       │
       ▼
70-Class Softmax
```

Training also uses class weighting, early stopping, model checkpointing, and learning-rate reduction.

---

## Application Architecture

```text
┌──────────────────────────────┐
│        Browser UI            │
│ HTML + CSS + JavaScript      │
└──────────────┬───────────────┘
               │ image upload
               ▼
┌──────────────────────────────┐
│        FastAPI API           │
│ /predict                     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ OpenCV Preprocessing         │
│ Line / Word / Segment        │
│ Extraction                   │
└──────────────┬───────────────┘
               │ 64×64 tensors
               ▼
┌──────────────────────────────┐
│ TensorFlow / Keras CNN       │
│ 70 writer classes            │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Prediction Aggregation       │
│ + Confidence Check           │
└──────────────────────────────┘
```

---

## Tech Stack

| Area | Technologies |
| --- | --- |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Numerical Processing | NumPy |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML5, CSS3, JavaScript |
| Model Format | Keras |
| Deployment | Docker / Hugging Face Spaces-compatible setup |

---

## Repository Structure

```text
.
├── backend/
│   └── app.py              # FastAPI inference service
├── frontend/
│   └── index.html          # Browser interface
├── train.py                # Training pipeline
├── run.py                  # Test/evaluation pipeline
├── model.keras             # Trained CNN model
├── labels.json             # 70 writer labels
├── requirements.txt
├── Dockerfile
└── README.md
```

The training and test image folders are intentionally excluded from the repository.

---

## Run the Application

### 1. Clone

```bash
git clone https://github.com/Leroy-laboe/handwriting-author-identification.git
cd handwriting-author-identification
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
uvicorn backend.app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

---

## Training

Expected training images are placed in `train_dir/`.

```bash
python train.py \
  --train_dir train_dir \
  --model_out model.keras \
  --labels_out labels.json
```

---

## Evaluation

Expected held-out evaluation images are placed in `test_dir/`.

```bash
python run.py \
  --test_dir test_dir \
  --model model.keras \
  --labels labels.json \
  --out result.csv
```

The evaluation script predicts the writer for each handwriting image, compares it with the writer ID encoded in the filename, and outputs the overall classification accuracy.

---

## What I Learned

This project gave me practical experience across the complete machine-learning application workflow:

- image preprocessing and segmentation;
- data augmentation;
- CNN architecture design;
- multi-class classification;
- model training and evaluation;
- converting an ML model into an inference API;
- connecting machine-learning inference to a user-facing web interface;
- packaging an application for deployment.

---

## Author

**Leroy Nyasha Mangwarara**

Computer Science · Data Science · Software Engineering · Applied AI

[GitHub](https://github.com/Leroy-laboe) · [LinkedIn](https://www.linkedin.com/in/leroy-nyasha-mangwarara-86185a302/) · [Email](mailto:mangwararaleroy@gmail.com)
