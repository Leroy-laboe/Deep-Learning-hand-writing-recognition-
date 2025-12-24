from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import json
from pathlib import Path
import logging

import keras
from keras.models import load_model

logging.basicConfig(level=logging.INFO)

# -------------------------------------------------
# Aggressive Keras Preprocessing Patch
# (Avoids ValueError: Unrecognized keyword arguments: {'value_range': ...})
# -------------------------------------------------
def patch_keras_preprocessing():
    try:
        from keras.src.layers.preprocessing.image_preprocessing import (
            random_rotation,
            random_translation,
            random_zoom,
            random_contrast,
        )

        layers_to_patch = [
            (random_rotation, "RandomRotation"),
            (random_translation, "RandomTranslation"),
            (random_zoom, "RandomZoom"),
            (random_contrast, "RandomContrast"),
        ]

        for module, class_name in layers_to_patch:
            cls = getattr(module, class_name)
            orig_init = cls.__init__

            def make_patched_init(old_init):
                def patched_init(self, *args, **kwargs):
                    kwargs.pop("value_range", None)
                    return old_init(self, *args, **kwargs)
                return patched_init

            cls.__init__ = make_patched_init(orig_init)

        logging.info("Successfully applied aggressive patches to Keras preprocessing layers.")
    except Exception as e:
        logging.warning(f"Failed to apply aggressive Keras patches: {e}")

patch_keras_preprocessing()


# -------------------------------------------------
# App setup
# -------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Paths (absolute, stable)
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.keras"
LABELS_PATH = BASE_DIR / "labels.json"

model = None
idx_to_label = {}

# -------------------------------------------------
# Load labels
# -------------------------------------------------
try:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    if isinstance(labels, dict):
        if all(str(k).isdigit() for k in labels.keys()):
            idx_to_label = {int(k): v for k, v in labels.items()}
        else:
            idx_to_label = {int(v): k for k, v in labels.items()}
    elif isinstance(labels, list):
        idx_to_label = {i: name for i, name in enumerate(labels)}
    else:
        raise TypeError(f"Unsupported labels.json type: {type(labels)}")

    logging.info(f"Loaded labels: {len(idx_to_label)}")

except Exception:
    logging.exception("Failed to load labels")


# -------------------------------------------------
# Load model
# -------------------------------------------------
try:
    logging.info(f"Loading model from: {MODEL_PATH}")
    model = load_model(str(MODEL_PATH), compile=False)
    logging.info("Model loaded successfully")
except Exception:
    logging.exception("Failed to load model")
    model = None


# -------------------------------------------------
# Preprocessing & Segmentation Helpers (from run.py)
# -------------------------------------------------
def to_grayscale(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # wait, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) is in run.py
    return img

# Correction from run.py:
def to_grayscale_real(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def segment_lines(gray):
    h = gray.shape[0]
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    proj = np.sum(closed, axis=1)
    if proj.max() == 0:
        return [(0, h)]
    thresh = max(1, int(0.03 * proj.max()))
    lines = []
    in_line = False
    start = 0
    for y, v in enumerate(proj):
        if v > thresh and not in_line:
            in_line = True
            start = y
        elif v <= thresh and in_line:
            end = y
            in_line = False
            if end - start >= 6:
                lines.append((max(0, start - 2), min(h, end + 2)))
    if in_line:
        lines.append((start, h))
    return lines


def segment_words_from_line(line_img):
    gray = to_grayscale_real(line_img)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(th, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 8 or h < 8:
            continue
        bboxes.append((x, y, w, h))
    bboxes = sorted(bboxes, key=lambda b: b[0])
    return [line_img[y:y + h, x:x + w] for (x, y, w, h) in bboxes]


def segment_chars_from_word(word_img, min_char_width=4):
    gray = to_grayscale_real(word_img)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cols = np.sum(th, axis=0)
    thresh = max(1, int(0.05 * cols.max()))
    separators = cols <= thresh
    chars = []
    in_char = False
    start = 0
    for i, val in enumerate(separators):
        if (not val) and (not in_char):
            in_char = True
            start = i
        elif val and in_char:
            end = i
            in_char = False
            if end - start >= min_char_width:
                chars.append(word_img[:, start:end])
    if in_char:
        end = len(separators)
        if end - start >= min_char_width:
            chars.append(word_img[:, start:end])
    return chars


def resize_and_normalize_char(ch_img, target_h, target_w):
    if len(ch_img.shape) == 2:
        ch = cv2.cvtColor(ch_img, cv2.COLOR_GRAY2RGB)
    else:
        ch = ch_img.copy()
    h, w = ch.shape[:2]
    scale = min(max(1e-6, target_w / w), max(1e-6, target_h / h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(ch, (nw, nh), interpolation=cv2.INTER_AREA)
    pad_left = (target_w - nw) // 2
    pad_top = (target_h - nh) // 2
    padded = 255 * np.ones((target_h, target_w, 3), dtype=np.uint8)
    padded[pad_top:pad_top + nh, pad_left:pad_left + nw, :] = resized
    return padded.astype(np.float32) / 255.0


@app.get("/health")
def health():
    return {
        "model_loaded": model is not None,
        "labels_count": len(idx_to_label),
        "model_path": str(MODEL_PATH),
        "labels_path": str(LABELS_PATH),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    content = await file.read()
    arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # -------------------------------------------------
    # Segmentation (matching run.py and training)
    # -------------------------------------------------
    gray = to_grayscale_real(img)
    lines = segment_lines(gray)
    chars_imgs = []

    for y1, y2 in lines:
        line_img = img[y1:y2, :]
        words = segment_words_from_line(line_img)
        if not words:
            words = [line_img]
        for w in words:
            chars = segment_chars_from_word(w, min_char_width=6)
            if not chars:
                # Fallback: if no characters found in word, use the whole word
                ch = resize_and_normalize_char(w, 64, 64)
                chars_imgs.append(ch)
            else:
                for c in chars:
                    ch = resize_and_normalize_char(c, 64, 64)
                    chars_imgs.append(ch)

    if not chars_imgs:
         raise HTTPException(
            status_code=422, 
            detail="No handwriting detected in the image. Please upload a clearer image of handwriting."
        )

    X_chars = np.array(chars_imgs)
    preds = model.predict(X_chars, verbose=0)
    
    # Get max probability for each character (confidence)
    char_confidences = np.max(preds, axis=1)
    avg_confidence = float(np.mean(char_confidences))
    
    # Threshold for rejection
    CONFIDENCE_THRESHOLD = 0.35
    
    if avg_confidence < CONFIDENCE_THRESHOLD:
         return {
            "author": "Unknown",
            "confidence": avg_confidence,
            "message": "Low confidence. This doesn't look like handwriting the model recognizes."
        }

    pred_indices = np.argmax(preds, axis=1)

    # Majority voting for the final author
    final_idx = int(np.bincount(pred_indices).argmax())
    return {
        "author": idx_to_label.get(final_idx, str(final_idx)),
        "confidence": avg_confidence
    }
