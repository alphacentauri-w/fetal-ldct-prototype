"""Low-dose CT simulation utilities."""

from __future__ import annotations

import cv2
import numpy as np


def simulate_low_dose_ct(
    image: np.ndarray,
    poisson_peak: float = 35.0,
    gaussian_sigma: float = 0.035,
    contrast_factor: float = 0.72,
    blur_sigma: float = 0.8,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate a low-dose CT image from a synthetic CT image.

    Args:
        image: Normalized synthetic CT image with values in [0, 1].
        poisson_peak: Photon-count scale for Poisson noise. Lower values create
            stronger quantum noise.
        gaussian_sigma: Standard deviation of additive Gaussian noise.
        contrast_factor: Multiplier used to reduce contrast around mid-gray.
        blur_sigma: Gaussian blur sigma that mimics mild resolution loss.
        seed: Optional random seed for reproducible experiments.

    Returns:
        Simulated normalized low-dose CT image with values in [0, 1].
    """

    rng = np.random.default_rng(seed)
    clipped = np.clip(image.astype(np.float32), 0.0, 1.0)

    poisson_noisy = rng.poisson(clipped * poisson_peak) / poisson_peak
    gaussian_noise = rng.normal(loc=0.0, scale=gaussian_sigma, size=clipped.shape)
    noisy = poisson_noisy + gaussian_noise

    low_contrast = 0.5 + contrast_factor * (noisy - 0.5)
    blurred = cv2.GaussianBlur(low_contrast.astype(np.float32), (0, 0), sigmaX=blur_sigma)

    return np.clip(blurred, 0.0, 1.0).astype(np.float32)
