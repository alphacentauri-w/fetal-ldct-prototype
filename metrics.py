"""Image quality metrics for prototype evaluation."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from skimage.metrics import structural_similarity


def calculate_mse(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Calculate mean squared error between two normalized images."""

    difference = reference.astype(np.float32) - candidate.astype(np.float32)
    return float(np.mean(np.square(difference)))


def calculate_psnr(reference: np.ndarray, candidate: np.ndarray, data_range: float = 1.0) -> float:
    """Calculate peak signal-to-noise ratio in decibels."""

    mse = calculate_mse(reference, candidate)
    if mse == 0.0:
        return math.inf
    return float(20.0 * math.log10(data_range / math.sqrt(mse)))


def calculate_ssim(reference: np.ndarray, candidate: np.ndarray, data_range: float = 1.0) -> float:
    """Calculate structural similarity index between two normalized images."""

    return float(structural_similarity(reference, candidate, data_range=data_range))


def calculate_metrics(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
    """Calculate MSE, PSNR, and SSIM for a candidate image.

    Args:
        reference: Reference image, typically synthetic CT.
        candidate: Candidate image, typically simulated low-dose CT.

    Returns:
        Dictionary suitable for display as a pandas table.
    """

    return {
        "MSE": calculate_mse(reference, candidate),
        "PSNR": calculate_psnr(reference, candidate),
        "SSIM": calculate_ssim(reference, candidate),
    }
