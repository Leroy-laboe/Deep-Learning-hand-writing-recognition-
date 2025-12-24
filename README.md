# ✍️ Professional Handwriting Identifier

A deep learning-based application that identifies the author of handwritten text with high precision. This project features a robust **FastAPI** backend and a modern, interactive **Glassmorphism** frontend.

![Project Status](https://img.shields.io/badge/Accuracy-83.57%25-success)
![Technology](https://img.shields.io/badge/Stack-Keras%20%7C%20FastAPI%20%7C%20OpenCV-blue)

## 🚀 Key Features

*   **High Accuracy:** Achieved **83.57%** accuracy on the test dataset.
*   **Intelligent Segmentation:** Automatically breaks down sentences into individual characters for per-character analysis.
*   **Majority Voting Algorithm:** Enhances reliability by collecting predictions from all characters/words in an image to determine the final author.
*   **Premium UX:** Modern, interactive frontend with animated background blobs and a predictive confidence progress bar.
*   **Reliability Filters:** Implemented a **0.35 confidence threshold** to reject non-handwriting or low-certainty images.

## 🛠️ Technology Stack

*   **Backend:** Python, FastAPI, Keras, TensorFlow, OpenCV
*   **Frontend:** HTML5, Modern CSS (Vanilla), JavaScript
*   **Preprocessing:** Grayscale conversion, Otsu thresholding, Line/Word/Character segmentation.

## 📦 Installation & Setup

1. **Download or Clone the project:**
   ```bash
   git clone https://github.com/Leroy-laboe/Deep-Learning-hand-writing-recognition-
   cd hand-writing-recognition
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the backend server:**
   ```bash
   uvicorn backend.app:app --reload
   ```

5. **Open the frontend:**
   Open `frontend/index.html` in your web browser.

## 📊 Performance Verification

To verify the model's accuracy on the test set, run:
```bash
python run.py --test_dir test_dir --model model.keras --labels labels.json
```

---
*Created as a Deep Learning Portfolio project focusing on Computer Vision and Handwriting Recognition.*
