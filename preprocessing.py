"""Utilities for preprocessing uploaded MRI slices and NIfTI tomograms.

The functions in this module intentionally keep preprocessing simple and
transparent for a prototype: convert an input image to grayscale, normalize the
intensity range, and resize it to a fixed square resolution expected by the
synthetic CT generation pipeline. NIfTI volumes are loaded as normalized 3D
arrays so that the Streamlit app can pass a selected 2D slice into the same
pipeline.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Tuple, Union

import cv2
import nibabel as nib
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

    image_float = np.nan_to_num(
        image.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    min_value = float(np.min(image_float))
    max_value = float(np.max(image_float))

    if np.isclose(max_value, min_value):
        return np.zeros_like(image_float, dtype=np.float32)

    return ((image_float - min_value) / (max_value - min_value)).astype(np.float32)


def load_nifti(file: Union[str, Path, BinaryIO]) -> np.ndarray:
    """Load a NIfTI tomogram and normalize it to the range [0, 1]."""

    try:
        if isinstance(file, (str, Path)):
            nifti_image = nib.load(str(file))
        else:
            file_name = str(getattr(file, "name", "")).lower()
            suffix = ".nii.gz" if file_name.endswith(".nii.gz") else ".nii"

            with NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
                file.seek(0)
                temporary_file.write(file.read())
                temporary_path = temporary_file.name

            nifti_image = nib.load(temporary_path)

    except Exception as exc:
        raise RuntimeError(f"Не удалось прочитать NIfTI-файл: {exc}") from exc

    volume = np.asanyarray(nifti_image.dataobj, dtype=np.float32)

    if volume.ndim == 4:
        volume = volume[..., 0]

    if volume.ndim != 3:
        raise ValueError(f"Ожидалась 3D-томограмма, получена размерность {volume.shape}.")

    return normalize_image(volume)


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


def preprocess_mri_slice(
    image: Union[Image.Image, np.ndarray],
    size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Run the complete MRI preprocessing pipeline.

    Args:
        image: Uploaded MRI slice in PNG/JPG format or a 2D NumPy slice from a
            NIfTI tomogram.
        size: Target output dimensions as ``(width, height)``.

    Returns:
        A normalized and resized grayscale image with values in [0, 1].
    """

    if isinstance(image, Image.Image):
        image_array = pil_to_grayscale_array(image)
    else:
        image_array = np.asarray(image, dtype=np.float32)
        if image_array.ndim != 2:
            raise ValueError(f"Ожидался 2D-срез, получена размерность {image_array.shape}.")

    normalized = normalize_image(image_array)
    return resize_image(normalized, size=size)
