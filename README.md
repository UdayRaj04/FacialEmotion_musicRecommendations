# 🎶 Webcam Facial-Emotion → Music Recommender (YouTube/Spotify)

A web application that detects facial emotions via webcam and recommends music tracks that match the user’s mood in real-time. Built with Flask, TensorFlow/Keras, OpenCV, and a React frontend.

🌐 **Live Website**: [https://facemoodmusic.onrender.com](https://facemoodmusic.onrender.com)

- Frontend: React (Vite)
- Backend: Flask (Python) + TensorFlow/Keras + OpenCV
- Dataset: Kaggle – Face Expression Recognition Dataset (FER) https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset
- Add DATASET - in backend/data/train/{angry,disgust,fear,happy,sad,surprise,neutral}/*.pngface-expression-recognition-dataset

## Features 

 - Real-time Emotion Detection: Detects 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral) using a CNN model.
 - Music Recommendations: Maps emotions to playlists from YouTube and Spotify via API.
 - Webcam Integration: Capture live images for instant emotion detection and music suggestions.
 - RESTful API: /predict (emotion from image), /recommend (music by emotion), /health (service status).
 - Training Enhancements: Data augmentation, model checkpoints, early stopping for better accuracy.

## Future Improvements

 - Real-time video emotion detection instead of single-frame captures.
 - Emotion-adaptive UI themes.
 - Improved model accuracy with larger datasets.

## Quickstart
### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Prepare dataset: ./data/train/{angry,disgust,fear,happy,sad,surprise,neutral}/*.png
python train.py --data-dir ./data --epochs 20 --batch-size 64 --val-split 0.15

python app.py   # runs http://localhost:5000
```

### Frontend
```bash
cd ../frontend
cp .env.sample .env   # optionally edit VITE_API_BASE
npm install
npm run dev           # opens http://localhost:5173
```

Open the app, allow webcam, click **Capture** → **Detect Emotion** → music recommendations appear.
