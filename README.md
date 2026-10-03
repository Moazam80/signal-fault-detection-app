# Intelligent Signal Fault Diagnostic System

Hackathon project — KNN vibration signal classifier with Gemini AI explanation layer.

## Project Structure

```
signal-fault-app/
+-- backend/
¦   +-- features.py          # Feature extraction (port of MATLAB extractFeatures)
¦   +-- signal_generator.py  # Synthetic signal generator (port of MATLAB generateSignal)
¦   +-- train_model.py       # KNN training (port of MATLAB Steps 1 + 4)
¦   +-- classifier.py        # Prediction + interpretation layer
¦   +-- agent.py             # Gemini AI explanation layer
¦   +-- main.py              # FastAPI REST API
¦   +-- requirements.txt
¦   +-- models/              # Saved trained model (auto-created)
+-- frontend/
¦   +-- index.html
¦   +-- style.css
¦   +-- app.js
+-- matlab_scripts/          # Original MATLAB scripts (for reference)
```

## Setup

### 1. Create and activate a virtual environment

```powershell
cd signal-fault-app\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Train the KNN model

```powershell
python train_model.py
```

This generates `models/knn_model.pkl`.

### 3. (Optional) Set your Gemini API key

```powershell
$env:GEMINI_API_KEY = "your-key-here"
```

### 4. Start the API server

```powershell
uvicorn main:app --reload
```

API runs at: http://localhost:8000
Interactive docs: http://localhost:8000/docs

### 5. Open the frontend

Open `frontend/index.html` in your browser,
OR visit http://localhost:8000/app (served by FastAPI).

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | /health | Health check |
| POST | /diagnose | Upload CSV/MAT ? get classification + AI explanation |
| GET | /generate-demo/{class} | Generate + classify a demo signal |

## Signal Classes

- Normal
- High Vibration
- Harmonic Fault
- Impulsive Fault
