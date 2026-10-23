"""Shared feature order for collection and recognition of mirrored frames."""
import numpy as np


def coordinates(result):
    data = np.zeros(126, dtype=np.float32)
    for hand, handedness in zip(result.hand_landmarks, result.handedness):
        offset = 0 if handedness[0].category_name == "Left" else 63
        data[offset:offset + 63] = [v for lm in hand for v in (lm.x, lm.y, lm.z)]
    return data
