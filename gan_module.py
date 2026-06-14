"""Prototype synthetic CT generation module.

This file mimics the public interface of a future GAN-based generator while
currently using deterministic intensity transformations. Replacing the body of
``generate_synthetic_ct`` with model inference should not require changes in the
Streamlit app or downstream LDCT simulation code.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage import exposure


def generate_synthetic_ct(image: np.ndarray) -> np.ndarray:
    """Generate a CT-like image from a preprocessed MRI slice.

    The current prototype applies contrast-limited histogram equalization,
    gamma correction, and a soft tissue/bone-inspired intensity remapping. This
    is not a clinically valid MRI-to-CT model; it is a transparent placeholder
    designed for a master's thesis prototype interface.

    Args:
        image: Normalized grayscale MRI slice with values in [0, 1].

    Returns:
        Synthetic CT-like image as ``float32`` in the range [0, 1].
    """

    clipped = np.clip(image.astype(np.float32), 0.0, 1.0)

    equalized = exposure.equalize_adapthist(clipped, clip_limit=0.03).astype(np.float32)
    gamma_corrected = exposure.adjust_gamma(equalized, gamma=0.75).astype(np.float32)

    # Emphasize bright anatomical boundaries to create a CT-like appearance.
    blurred = cv2.GaussianBlur(gamma_corrected, (0, 0), sigmaX=1.2)
    detail = cv2.addWeighted(gamma_corrected, 1.35, blurred, -0.35, 0.0)

    # Piecewise remapping approximates CT windowing behavior for visualization.
    synthetic_ct = np.where(
        detail < 0.35,
        detail * 0.55,
        0.20 + np.power(detail, 1.8) * 0.80,
    )

    return np.clip(synthetic_ct, 0.0, 1.0).astype(np.float32)
