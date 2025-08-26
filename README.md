# 🎶 Webcam Facial-Emotion → Music Recommender (YouTube/Spotify)

- Frontend: React (Vite)
- Backend: Flask (Python) + TensorFlow/Keras + OpenCV
- Dataset: Kaggle – Face Expression Recognition Dataset (FER) https://www.kaggle.com/datasets/jonathanoheix/
- Add DATASET - in backend/data/train/{angry,disgust,fear,happy,sad,surprise,neutral}/*.pngface-expression-recognition-dataset

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
