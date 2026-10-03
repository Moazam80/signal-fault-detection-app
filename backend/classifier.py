"""
classifier.py
-------------
Port of MATLAB Step 5B logic - prediction + interpretation layer.

Loads the trained KNN model and exposes a single classify() function
that takes a raw signal array and returns a full diagnostic result dict.
"""

import pickle
from pathlib import Path

import numpy as np

from features import extract_features

import sys
MODEL_PATH = (Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).parent) / "models" / "knn_model.pkl"

# Cached model bundle (loaded once on first call)
_bundle = None


def _load_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run train_model.py first."
            )
        with open(MODEL_PATH, "rb") as f:
            _bundle = pickle.load(f)
    return _bundle


def classify(x: np.ndarray, fs: float) -> dict:
    """
    Classify a vibration signal and return a full diagnostic result.

    Parameters
    ----------
    x  : 1-D numpy signal array
    fs : sampling frequency (Hz)

    Returns
    -------
    dict with:
        predicted_class   (str)
        scores            (dict: class -> vote fraction)
        features          (dict of extracted feature values)
        interpretation    (dict: observations list + summary text)
        warnings          (list of warning strings)
    """
    bundle = _load_bundle()
    knn    = bundle["knn"]
    scaler = bundle["scaler"]
    classes = bundle["classes"]
    fundamental_range = bundle["fundamental_range"]

    # --- Validation ---
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if len(x) < 10:
        raise ValueError("Signal is too short or contains no valid samples.")

    # --- Feature extraction ---
    feats = extract_features(x, fs, fundamental_range)
    feat_vector = feats["vector"].reshape(1, -1)

    # --- Scale using training statistics ---
    feat_scaled = scaler.transform(feat_vector)

    # --- KNN prediction ---
    predicted_class = knn.predict(feat_scaled)[0]

    # Vote fractions as "scores" (equivalent to MATLAB knn scores)
    proba = knn.predict_proba(feat_scaled)[0]
    scores = {cls: float(p) for cls, p in zip(knn.classes_, proba)}

    # --- Warnings ---
    warnings = []
    if fs <= 140:
        warnings.append(
            "Sampling frequency is relatively low for the 40-70 Hz model."
        )
    f_fund = feats["fundamental_freq"]
    if f_fund < 40 or f_fund > 70:
        warnings.append(
            f"Detected fundamental ({f_fund:.1f} Hz) is outside the "
            "40-70 Hz training range. Prediction may be unreliable."
        )

    # --- Interpretation layer (port of MATLAB switch block) ---
    interpretation = _interpret(predicted_class, feats)

    return {
        "predicted_class": predicted_class,
        "scores": scores,
        "features": {
            "rms":                   round(feats["rms"],                 4),
            "peak":                  round(feats["peak"],                4),
            "crest_factor":          round(feats["crest_factor"],        4),
            "fundamental_freq_hz":   round(feats["fundamental_freq"],    2),
            "second_harmonic_ratio": round(feats["second_harmonic_ratio"], 4),
        },
        "interpretation": interpretation,
        "warnings": warnings,
    }


# -----------------------------------------------------------------------
# Interpretation layer  (port of MATLAB diagnostic switch block)
# -----------------------------------------------------------------------

def _interpret(predicted_class: str, feats: dict) -> dict:
    cls      = predicted_class.lower().strip()
    rms      = feats["rms"]
    peak     = feats["peak"]
    cf       = feats["crest_factor"]
    shr      = feats["second_harmonic_ratio"]
    f_fund   = feats["fundamental_freq"]

    observations = []
    summary      = ""

    if cls == "normal":
        observations.append(
            "RMS level is within normal range." if rms < 1.0
            else "RMS level is slightly elevated; monitor for changes."
        )
        observations.append(
            "Crest Factor is low - no strong impulsive behaviour." if cf < 2.5
            else "Crest Factor is elevated - review for transient events."
        )
        observations.append(
            "Second Harmonic Ratio is low - limited harmonic distortion." if shr < 0.10
            else "Second Harmonic Ratio is elevated - harmonic content present."
        )
        summary = (
            "Features are consistent with a healthy machine. "
            "No dominant impulsive or harmonic pattern detected."
        )

    elif cls == "high vibration":
        observations.append(
            "RMS is elevated - increased overall vibration magnitude." if rms >= 1.0
            else "RMS is not strongly elevated despite the High Vibration classification."
        )
        observations.append(
            "Peak amplitude is elevated." if peak >= 2.0
            else "Peak amplitude is not strongly elevated."
        )
        observations.append(
            "Crest Factor is moderate - signal is not dominated by sharp impulses." if cf < 2.5
            else "Crest Factor is elevated - transient behaviour may also be present."
        )
        observations.append(
            "Second Harmonic Ratio is low - harmonic distortion is not dominant." if shr < 0.10
            else "Second Harmonic Ratio is elevated - harmonic content also present."
        )
        summary = (
            "Classification driven by increased overall vibration magnitude. "
            "Not primarily an impulsive or harmonic pattern."
        )

    elif cls == "harmonic fault":
        observations.append(
            "Second Harmonic Ratio is elevated - significant 2nd harmonic content." if shr >= 0.10
            else "Second Harmonic Ratio is not strongly elevated; check spectrum for other harmonics."
        )
        observations.append(
            "Crest Factor is moderate - not dominated by sharp impulses." if cf < 2.5
            else "Crest Factor is elevated - some transient behaviour may also be present."
        )
        observations.append(
            f"Fundamental frequency is {f_fund:.1f} Hz - within the model operating range."
        )
        summary = (
            "Signal contains a harmonic pattern. "
            "The second harmonic is the primary diagnostic indicator. "
            "Possible causes: misalignment, looseness, or non-linear stiffness."
        )

    elif cls == "impulsive fault":
        observations.append(
            "Crest Factor is high - strong impulsive / transient behaviour." if cf >= 3.0
            else "Crest Factor is not strongly elevated; the impulsive signature is relatively weak."
        )
        observations.append(
            "Peak amplitude is elevated - strong instantaneous event detected." if peak >= 2.0
            else "Peak amplitude is not strongly elevated."
        )
        observations.append(
            "Second Harmonic Ratio is low - harmonic distortion is not dominant." if shr < 0.10
            else "Second Harmonic Ratio is elevated - harmonic content also present."
        )
        summary = (
            "Combination of high peak and elevated Crest Factor is consistent "
            "with an impulsive vibration pattern. "
            "Possible causes: bearing defect, gear tooth impact, or mechanical looseness."
        )

    else:
        summary = "No specific interpretation rule available for this class."

    return {"observations": observations, "summary": summary}

