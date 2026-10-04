import io, tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.io
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from classifier import classify
from agent import get_ai_explanation
from signal_generator import generate_signal, CLASSES

app = FastAPI(title="Signal Fault Diagnostic API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

import sys
FRONTEND_PATH = (Path(sys._MEIPASS) / "frontend") if getattr(sys, "frozen", False) else (Path(__file__).parent.parent / "frontend")
if FRONTEND_PATH.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_PATH), html=True), name="frontend")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...), fs: Optional[float] = Form(None)):
    ext = Path(file.filename or "").suffix.lower()
    contents = await file.read()
    try:
        x, fs = _read_csv(contents, fs) if ext == ".csv" else _read_mat(contents, fs) if ext == ".mat" else (_ for _ in ()).throw(HTTPException(400, f"Unsupported: {ext}"))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"Could not read file: {e}")
    try:
        result = classify(x, fs)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    result["ai"] = get_ai_explanation(result)
    result["signal_info"] = {"n_samples": len(x), "fs_hz": fs, "duration_s": round(len(x)/fs, 3)}
    return JSONResponse(content=result)


@app.get("/generate-demo/{class_name}")
def generate_demo(class_name: str):
    cn = class_name.replace("+", " ").replace("%20", " ").strip()
    if cn not in CLASSES:
        raise HTTPException(400, f"Unknown class. Valid: {CLASSES}")
    t = np.arange(0, 2.0, 0.001)
    x = generate_signal(cn, t)
    result = classify(x, 1000.0)
    result["ai"] = get_ai_explanation(result)
    result["signal_info"] = {"n_samples": len(x), "fs_hz": 1000.0, "duration_s": 2.0, "true_class": cn}
    return JSONResponse(content=result)


@app.get("/generate-signal/{class_name}")
def generate_signal_csv(class_name: str):
    cn = class_name.replace("+", " ").replace("%20", " ").strip()
    if cn not in CLASSES:
        raise HTTPException(400, f"Unknown class. Valid: {CLASSES}")
    t = np.arange(0, 2.0, 0.001)
    x = generate_signal(cn, t)
    buf = io.StringIO()
    buf.write("Time,Acceleration\n")
    for ti, xi in zip(t, x):
        buf.write(f"{ti:.6f},{xi:.8f}\n")
    buf.seek(0)
    safe = cn.replace(" ", "_")
    return StreamingResponse(io.BytesIO(buf.getvalue().encode()), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={safe}_Signal.csv"})


def _read_csv(contents: bytes, fs):
    first_line = contents.decode(errors="replace").split("\n")[0].strip()
    has_header = False
    try:
        [float(v) for v in first_line.split(",") if v.strip()]
    except ValueError:
        has_header = True
    df = pd.read_csv(io.BytesIO(contents), header=0 if has_header else None)
    data = df.dropna(how="all").to_numpy(dtype=float)
    data = data[:, np.any(np.isfinite(data), axis=0)]
    if data.shape[1] == 1:
        x = data[:, 0]
        if not fs or fs <= 0:
            raise HTTPException(422, "1-column CSV detected. Please provide Fs (Hz).")
    elif data.shape[1] >= 2:
        t_col, x = data[:, 0], data[:, 1]
        ok = np.isfinite(t_col) & np.isfinite(x)
        t_col, x = t_col[ok], x[ok]
        if len(t_col) < 2:
            raise HTTPException(422, "Not enough valid samples.")
        dt = float(np.median(np.diff(t_col)))
        if dt <= 0:
            raise HTTPException(422, "Time column must be increasing.")
        fs = 1.0 / dt
    else:
        raise HTTPException(422, "Cannot parse CSV structure.")
    return x, fs


def _read_mat(contents: bytes, fs):
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        tmp.write(contents); tmp_path = tmp.name
    try:
        mat = scipy.io.loadmat(tmp_path)
    except Exception as e:
        raise HTTPException(422, f"Cannot read MAT file: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    cands = [k for k in mat if not k.startswith("__") and isinstance(mat[k], np.ndarray) and mat[k].squeeze().ndim == 1]
    if not cands:
        raise HTTPException(422, "No numeric 1-D vector found in MAT file.")
    x = mat[cands[0]].squeeze().astype(float)
    if not fs or fs <= 0:
        raise HTTPException(422, "Please provide Fs (Hz) for MAT files.")
    return x, fs
