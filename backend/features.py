"""
features.py
-----------
Python port of the MATLAB extractFeatures() function.

Extracts 5 features from a vibration signal:
    1. RMS
    2. Peak
    3. Crest Factor
    4. Fundamental Frequency  (searched in [f_low, f_high] Hz)
    5. Second Harmonic Ratio

Mathematically identical to the MATLAB version.
"""

import numpy as np


def extract_features(
    x: np.ndarray,
    fs: float,
    fundamental_range: tuple = (40.0, 70.0),
) -> dict:
    """
    Extract the 5 diagnostic features from a 1-D vibration signal.

    Parameters
    ----------
    x                 : 1-D signal array
    fs                : sampling frequency (Hz)
    fundamental_range : (low, high) Hz band to search for fundamental

    Returns
    -------
    dict with keys: rms, peak, crest_factor, fundamental_freq,
                    second_harmonic_ratio, vector (numpy array in training order)
    """
    x = np.asarray(x, dtype=float).ravel()

    # 1. RMS
    rms = float(np.sqrt(np.mean(x ** 2)))

    # 2. Peak
    peak = float(np.max(np.abs(x)))

    # 3. Crest Factor
    crest_factor = peak / rms if rms > 0 else 0.0

    # 4. FFT - one-sided amplitude spectrum (identical to MATLAB convention)
    n = len(x)
    X = np.fft.fft(x)
    p2 = np.abs(X / n)
    p1 = p2[: n // 2 + 1].copy()
    if len(p1) > 2:
        p1[1:-1] *= 2
    f = fs * np.arange(len(p1)) / n

    # 5. Fundamental frequency - search inside operating band only
    f_low, f_high = fundamental_range
    band_mask = (f >= f_low) & (f <= f_high)

    if not np.any(band_mask):
        raise ValueError(
            f"Fs={fs} Hz too low to resolve {f_low}-{f_high} Hz fundamental band."
        )

    band_spectrum    = p1[band_mask]
    band_freqs       = f[band_mask]
    local_idx        = int(np.argmax(band_spectrum))
    fundamental_amp  = float(band_spectrum[local_idx])
    fundamental_freq = float(band_freqs[local_idx])

    # 6. Second Harmonic Ratio  (search +-2 Hz around 2x fundamental)
    second_harmonic_freq = 2.0 * fundamental_freq
    harmonic_mask = np.abs(f - second_harmonic_freq) <= 2.0

    if np.any(harmonic_mask):
        second_harmonic_amp   = float(np.max(p1[harmonic_mask]))
        second_harmonic_ratio = second_harmonic_amp / max(fundamental_amp, 1e-12)
    else:
        second_harmonic_ratio = 0.0

    return {
        "rms":                   rms,
        "peak":                  peak,
        "crest_factor":          crest_factor,
        "fundamental_freq":      fundamental_freq,
        "second_harmonic_ratio": second_harmonic_ratio,
        # flat vector in training order - used by the classifier
        "vector": np.array(
            [rms, peak, crest_factor, fundamental_freq, second_harmonic_ratio],
            dtype=float,
        ),
    }

