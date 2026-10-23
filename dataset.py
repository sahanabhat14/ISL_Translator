"""Validation for the initial two-hand, 30-frame baseline."""
from pathlib import Path
import numpy as np

SIGNS = ["HELLO", "THANK_YOU", "YES", "NO", "BYE", "NO_SIGN"]
FRAMES, FEATURES = 30, 126


def validate(sequence):
    if sequence.shape != (FRAMES, FEATURES):
        raise ValueError(f"expected {(FRAMES, FEATURES)}, got {sequence.shape}")
    if not np.isfinite(sequence).all():
        raise ValueError("contains NaN or infinite values")
    coverage = np.any(sequence != 0, axis=1).mean()
    if coverage < 0.8:
        raise ValueError(f"hands detected in only {coverage:.0%} of frames (need 80%)")


def load_dataset(root):
    samples, labels, paths, rejected = [], [], [], []
    for label, sign in enumerate(SIGNS):
        for path in sorted((Path(root) / sign).glob("*.npy")):
            try:
                sample = np.load(path, allow_pickle=False).astype(np.float32)
                validate(sample)
            except (ValueError, OSError, TypeError) as error:
                rejected.append(f"{path}: {error}")
                continue
            samples.append(sample)
            labels.append(label)
            paths.append(str(path))
    return np.asarray(samples, dtype=np.float32), np.asarray(labels, dtype=int), paths, rejected
