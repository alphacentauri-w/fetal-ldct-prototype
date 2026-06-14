"""Utilities for preprocessing uploaded MRI slices.

The functions in this module intentionally keep preprocessing simple and
transparent for a prototype: convert an input image to grayscale, normalize the
intensity range, and resize it to a fixed square resolution expected by the
synthetic CT generation pipeline.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image


DEFAULT_IMAGE_SIZE: Tuple[int, int] = (256, 256)


def pil_to_grayscale_array(image: Image.Image) -> np.ndarray:
    """Convert a Pillow image to a 2D grayscale NumPy array.

    Args:
        image: Input MRI slice loaded with Pillow.

    Returns:
        A two-dimensional ``uint8`` array with values in the range [0, 255].
    """

    grayscale_image = image.convert("L")
    return np.asarray(grayscale_image, dtype=np.uint8)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize an image to floating-point intensities in the range [0, 1].

    Args:
        image: Input grayscale image as a NumPy array.

    Returns:
        A ``float32`` normalized image. Constant images are converted to zeros to
        avoid division by zero.
    """

    image_float = image.astype(np.float32)
    min_value = float(np.min(image_float))
    max_value = float(np.max(image_float))

    if np.isclose(max_value, min_value):
        return np.zeros_like(image_float, dtype=np.float32)

    return ((image_float - min_value) / (max_value - min_value)).astype(np.float32)


def resize_image(image: np.ndarray, size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Resize an image to the configured model input size.

    Args:
        image: Input image as a two-dimensional array.
        size: Target size as ``(width, height)`` for OpenCV.

    Returns:
        Resized image with ``float32`` dtype.
    """

    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def preprocess_mri_slice(image: Image.Image, size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Run the complete MRI preprocessing pipeline.

    Args:
        image: Uploaded MRI slice in PNG or JPG format.
        size: Target output dimensions as ``(width, height)``.

    Returns:
        A normalized and resized grayscale image with values in [0, 1].
    """

    grayscale = pil_to_grayscale_array(image)
    normalized = normalize_image(grayscale)
    return resize_image(normalized, size=size)
