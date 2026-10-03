"""
signal_generator.py
-------------------
Python port of the MATLAB generateVibration() / generateSignal() functions.

Used to:
  - Generate synthetic signals for training the KNN model
  - Provide test signals for the demo endpoint

Signal classes:
    Normal         - sinusoid + small noise
    High Vibration - amplitude-scaled sinusoid
    Harmonic Fault - sinusoid + second harmonic component
    Impulsive Fault- sinusoid + short sharp impulse
"""

import numpy as np


CLASSES = ["Normal", "High Vibration", "Harmonic Fault", "Impulsive Fault"]


def generate_signal(
    class_name: str,
    t: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate one synthetic vibration signal.

    Parameters
    ----------
    class_name : one of CLASSES
    t          : time vector (seconds)
    rng        : numpy random generator (pass one for reproducibility)

    Returns
    -------
    x : 1-D numpy array, same length as t
    """
    if rng is None:
        rng = np.random.default_rng()

    # Random operating conditions - identical distributions to MATLAB
    base_frequency = 40.0 + 30.0 * rng.random()       # 40-70 Hz
    amplitude      = 0.8  + 0.4  * rng.random()
    phase          = 2.0  * np.pi * rng.random()
    noise_level    = 0.02 + 0.08 * rng.random()

    # Base sinusoid
    x = amplitude * np.sin(2.0 * np.pi * base_frequency * t + phase)

    if class_name == "Normal":
        pass  # nothing extra

    elif class_name == "High Vibration":
        vibration_multiplier = 1.5 + 1.2 * rng.random()
        x = vibration_multiplier * x

    elif class_name == "Harmonic Fault":
        harmonic_amplitude = 0.25 + 0.60 * rng.random()
        second_harmonic = (
            harmonic_amplitude * amplitude
            * np.sin(2.0 * np.pi * (2.0 * base_frequency) * t + phase)
        )
        x = x + second_harmonic

    elif class_name == "Impulsive Fault":
        impulse_amplitude = 2.5 + 3.5 * rng.random()
        impulse_length    = rng.integers(2, 9)           # 2-8 samples
        impulse_start     = rng.integers(100, len(t) - impulse_length - 100)

        impulse = np.zeros_like(t)
        impulse[impulse_start : impulse_start + impulse_length] = (
            impulse_amplitude * (0.7 + 0.3 * rng.random())
        )
        x = x + impulse

    else:
        raise ValueError(f"Unknown class: {class_name}. Choose from {CLASSES}")

    # Measurement noise
    x = x + noise_level * rng.standard_normal(len(t))

    return x

