# ImageGuard Backend

This backend serves the trained CIFAKE EfficientNetB0 model used by ImageGuard.

## Model

- Architecture: EfficientNetB0
- Input: 224 × 224 RGB
- Output: probability of AI-GENERATED
- REAL: probability < 0.5
- AI-GENERATED: probability >= 0.5
- Grad-CAM: generated from EfficientNetB0 `top_activation`

## Run on Windows PowerShell

From this `backend` folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

The API will run at:

`http://127.0.0.1:5000`

Health check:

`http://127.0.0.1:5000/api/health`

Analysis endpoint:

`POST http://127.0.0.1:5000/api/analyze`

Field name:

`image`

## Frontend

Open `../frontend/index.html` with VS Code Live Server, or serve the frontend from another local web server. The frontend is configured to call `http://127.0.0.1:5000/api/analyze`.

The UI does not use mock predictions anymore. If the backend is unavailable, the interface shows a backend error instead of fabricating a result.
