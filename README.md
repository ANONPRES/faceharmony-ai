# FaceHarmony AI

Educational facial geometry analyzer. Upload a photo to evaluate measurable proportions, symmetry, and harmonic balance using MediaPipe Face Landmarker (478 landmarks), OpenCV, and NumPy.

**This app does not measure beauty.** Scores describe geometric relationships only.

## Stack

| Layer | Tech |
|-------|------|
| Frontend | Next.js 15, TypeScript, Tailwind CSS v4, Framer Motion, Recharts, Lucide React |
| Backend | FastAPI, MediaPipe Face Landmarker, OpenCV, NumPy |
| Storage | Browser `localStorage` for analysis history |

## Project structure

```
faceharmony-ai/
├── backend/
│   ├── app/
│   │   ├── api/routes.py          # POST /analyze, GET /health
│   │   ├── services/              # Landmarks, metrics, recommendations
│   │   ├── utils/image.py
│   │   ├── schemas.py
│   │   └── main.py
│   ├── models/face_landmarker.task
│   ├── requirements.txt
│   └── venv/
└── frontend/
    └── src/
        ├── app/                   # Pages: home, upload, results, history
        ├── components/            # UI building blocks
        └── lib/                   # API client, history, types
```

## Prerequisites

- Node.js 18+ and npm
- Python 3.11–3.13
- ~4 MB free for the MediaPipe model (`backend/models/face_landmarker.task`)

## Backend setup

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

If `models/face_landmarker.task` is missing, download it:

```bash
# PowerShell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" `
  -OutFile "models/face_landmarker.task"

# curl
mkdir -p models
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Start the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  

### `POST /analyze`

Multipart form field: `file` (JPEG / PNG / WEBP / BMP).

Example response shape:

```json
{
  "overall": 89,
  "symmetry": 92,
  "golden_ratio": 86,
  "eyes": 91,
  "nose": 82,
  "lips": 88,
  "jaw": 84,
  "chin": 90,
  "thirds": 87,
  "fifths": 85,
  "recommendations": ["..."],
  "metrics": {},
  "landmarks": [],
  "overlay": {}
}
```

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

Optional env (defaults to `http://localhost:8000`):

```bash
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Scoring weights

| Metric | Weight |
|--------|--------|
| Symmetry | 25% |
| Golden ratio | 15% |
| Facial thirds | 15% |
| Facial fifths | 10% |
| Jaw definition | 10% |
| Eyes | 10% |
| Nose | 5% |
| Lips | 5% |
| Chin | 5% |

Every metric is normalized to **0–100**. The overall harmony score is the weighted sum.

## Features

1. **Home** — dark glassmorphism landing with CTA  
2. **Upload** — drag & drop, file picker, camera capture, validation  
3. **Analysis** — MediaPipe landmarks + proportion metrics  
4. **Results** — animated score ring, breakdown cards, radar chart, suggestions  
5. **Overlay** — toggle landmarks, symmetry line, thirds, fifths, golden guides, measurements  
6. **History** — local save + compare two analyses  

## Notes

- Use a clear, front-facing photo with even lighting for the most stable readings.
- Camera angle, expression, and lens distortion can shift geometric scores.
- History is stored only in the current browser; clearing site data removes it.

## License

MIT — for educational and demonstration use.
