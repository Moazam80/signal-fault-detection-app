"""
train_model.py
--------------
Python port of MATLAB Steps 1 + 4.

Generates synthetic training / testing data, trains a KNN classifier
(K=5, euclidean, z-score scaled), and saves the model to models/knn_model.pkl.

Run once before starting the API server:
    python train_model.py
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from features import extract_features
from signal_generator import generate_signal, CLASSES

# -----------------------------------------------------------------------
# Settings  (matches MATLAB Step 1)
# -----------------------------------------------------------------------
FS               = 1000        # sampling frequency (Hz)
DURATION         = 2           # seconds
FUNDAMENTAL_RANGE = (40.0, 70.0)
N_TRAIN_PER_CLASS = 700
N_TEST_PER_CLASS  = 300
RANDOM_SEED       = 42
MODEL_PATH        = Path(__file__).parent / "models" / "knn_model.pkl"


def build_dataset(n_per_class: int, rng: np.random.Generator):
    t = np.arange(0, DURATION, 1.0 / FS)
    X, y = [], []
    for class_name in CLASSES:
        for _ in range(n_per_class):
            signal   = generate_signal(class_name, t, rng)
            features = extract_features(signal, FS, FUNDAMENTAL_RANGE)
            X.append(features["vector"])
            y.append(class_name)
    return np.array(X), np.array(y)


def train_and_save():
    print("=" * 50)
    print("   SIGNAL FAULT KNN TRAINING")
    print("=" * 50)

    rng = np.random.default_rng(RANDOM_SEED)

    print(f"\nGenerating {N_TRAIN_PER_CLASS * len(CLASSES)} training samples...")
    X_train, y_train = build_dataset(N_TRAIN_PER_CLASS, rng)

    print(f"Generating {N_TEST_PER_CLASS  * len(CLASSES)} testing  samples...")
    X_test,  y_test  = build_dataset(N_TEST_PER_CLASS,  rng)

    # Shuffle
    train_idx = rng.permutation(len(X_train))
    test_idx  = rng.permutation(len(X_test))
    X_train, y_train = X_train[train_idx], y_train[train_idx]
    X_test,  y_test  = X_test[test_idx],   y_test[test_idx]

    # Z-score scale using ONLY training statistics (same as MATLAB)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # KNN  (K=5, euclidean - identical to MATLAB fitcknn settings)
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(X_train_scaled, y_train)

    # Evaluate on unseen test set
    y_pred = knn.predict(X_test_scaled)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy : {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES))
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_test, y_pred, labels=CLASSES))

    # Save model + scaler
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "knn":     knn,
        "scaler":  scaler,
        "classes": CLASSES,
        "fs":      FS,
        "fundamental_range": FUNDAMENTAL_RANGE,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nModel saved to: {MODEL_PATH}")
    print("=" * 50)
    print("Training complete. Start the API server with: uvicorn main:app --reload")
    print("=" * 50)


if __name__ == "__main__":
    train_and_save()

