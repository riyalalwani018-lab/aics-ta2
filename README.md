# ImageGuard — Integrated AI Image Detection Project

ImageGuard uses the trained CIFAKE EfficientNetB0 model to classify uploaded images as REAL or AI-GENERATED.

## Structure

```text
ImageGuard/
├── frontend/
│   └── index.html
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── requirements.txt
│   ├── run_backend.ps1
│   ├── README.md
│   ├── metrics.json
│   └── models/
│       └── imageguard_efficientnetb0.keras
└── README.md
```

## Model test metrics

- Accuracy: 96.93%
- Precision: 97.21%
- Recall: 96.63%
- F1: 96.92%
- ROC-AUC: 99.60%

## Run

1. Start the backend from `backend`:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

2. Open `frontend/index.html` with VS Code Live Server.

3. Upload a JPG, PNG, or WEBP image.

4. Click **ANALYZE**.

The frontend sends the image to Flask, Flask runs the actual trained model, and the existing ImageGuard report is populated with the prediction, confidence, processing time, and Grad-CAM overlay.
