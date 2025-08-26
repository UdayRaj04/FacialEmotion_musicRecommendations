
import io
import os
import json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

app = Flask(__name__)
CORS(app)

EMOTIONS = ["angry","disgust","fear","happy","sad","surprise","neutral"]

# MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_best.keras")
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

_model = None

def load_emotion_model():
    global _model
    if _model is None:
        if load_model is None:
            raise RuntimeError("TensorFlow/Keras not available. Please install requirements and restart.")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("model.h5 not found. Train with train.py first.")
        _model = load_model(MODEL_PATH)
    return _model

def preprocess_image_for_fer(pil_img):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(48,48))

    if len(faces) == 0:
        h, w = gray.shape
        size = min(h, w)
        y0 = (h - size)//2
        x0 = (w - size)//2
        face_roi = gray[y0:y0+size, x0:x0+size]
    else:
        x, y, w, h = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
        face_roi = gray[y:y+h, x:x+w]

    face_roi = cv2.resize(face_roi, (48,48), interpolation=cv2.INTER_AREA)
    face_roi = face_roi.astype("float32") / 255.0
    face_roi = np.expand_dims(face_roi, axis=-1)
    face_roi = np.expand_dims(face_roi, axis=0)
    return face_roi

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/predict")
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file (field name 'image')."}), 400
        file = request.files["image"]
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        x = preprocess_image_for_fer(pil_img)
        model = load_emotion_model()
        probs = model.predict(x, verbose=0)[0]
        best_idx = int(np.argmax(probs))
        return jsonify({
            "emotion": EMOTIONS[best_idx],
            "scores": {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}
        })
    except (FileNotFoundError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500

def load_emotion_map():
    with open(os.path.join(os.path.dirname(__file__), "emotion_map.json"), "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/recommend")
def recommend():
    emotion = request.args.get("emotion", "neutral").lower().strip()
    source = request.args.get("source", "youtube").lower().strip()
    if emotion not in EMOTIONS:
        emotion = "neutral"
    data = load_emotion_map()
    items = data.get(emotion, {}).get(source, [])
    return jsonify({"emotion": emotion, "source": source, "items": items})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
