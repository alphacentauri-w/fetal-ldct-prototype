"""Streamlit prototype for MRI-to-synthetic-LDCT fetal image generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from gan_module import generate_synthetic_ct
from ldct_simulator import simulate_low_dose_ct
from metrics import calculate_metrics
from preprocessing import load_nifti, preprocess_mri_slice

OUTPUT_DIR = Path("outputs")


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a normalized floating-point image to ``uint8`` for display/saving."""

    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def get_slice_count(volume_shape: Tuple[int, int, int], axis: int) -> int:
    """Return the number of available 2D slices along a NIfTI axis."""

    return int(volume_shape[axis])


def extract_nifti_slice(volume: np.ndarray, axis: int, slice_index: int) -> np.ndarray:
    """Extract a 2D slice from a normalized 3D NIfTI volume."""

    if volume.ndim != 3:
        raise ValueError(f"Ожидалась 3D-томограмма, получена размерность {volume.shape}.")

    if axis == 0:
        return volume[slice_index, :, :]
    if axis == 1:
        return volume[:, slice_index, :]
    if axis == 2:
        return volume[:, :, slice_index]

    raise ValueError(f"Неподдерживаемая ось среза: {axis}.")


def save_results(
    original_mri: np.ndarray,
    synthetic_ct: np.ndarray,
    synthetic_ldct: np.ndarray,
    metrics: Dict[str, float],
) -> Path:
    """Save generated images and metrics to a timestamped output folder."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = OUTPUT_DIR / f"result_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(image_to_uint8(original_mri)).save(result_dir / "preprocessed_mri.png")
    Image.fromarray(image_to_uint8(synthetic_ct)).save(result_dir / "synthetic_ct.png")
    Image.fromarray(image_to_uint8(synthetic_ldct)).save(result_dir / "synthetic_ldct.png")
    pd.DataFrame([metrics]).to_csv(result_dir / "metrics.csv", index=False)

    return result_dir


def main() -> None:
    """Run the Streamlit user interface."""

    st.set_page_config(page_title="MRI → Synthetic LDCT Prototype", layout="wide")
    st.title("Прототип генерации синтетических НДКТ-подобных изображений плода")
    st.markdown(
        "Загрузите МРТ-срез в формате PNG/JPG или медицинскую томограмму "
        "NIfTI (.nii, .nii.gz). Для NIfTI отображается количество срезов томограммы."
    )

    uploaded_file = st.file_uploader(
        "Загрузите МРТ-срез или томограмму",
        type=["png", "jpg", "jpeg", "nii", "nii.gz"],
    )

    with st.sidebar:
        st.header("Параметры НДКТ")
        poisson_peak = st.slider("Интенсивность сигнала Пуассона", 10.0, 80.0, 35.0, 5.0)
        gaussian_sigma = st.slider("σ гауссова шума", 0.0, 0.10, 0.035, 0.005)
        contrast_factor = st.slider("Коэффициент контраста", 0.40, 1.00, 0.72, 0.02)
        blur_sigma = st.slider("σ гауссова размытия", 0.0, 2.0, 0.8, 0.1)
        random_seed = st.number_input("Seed", min_value=0, max_value=999_999, value=42, step=1)

    if uploaded_file is None:
        st.info("Ожидается загрузка изображения для запуска прототипа.")
        return

    file_name = uploaded_file.name.lower()
    is_nifti = file_name.endswith((".nii", ".nii.gz"))

    try:
        if is_nifti:
            volume = load_nifti(uploaded_file)
            axis_labels = {
                "Аксиальная (ось Z)": 2,
                "Корональная (ось Y)": 1,
                "Сагиттальная (ось X)": 0,
            }
            axis_label = st.selectbox(
                "Плоскость среза",
                options=list(axis_labels.keys()),
                help="Выберите направление, в котором будет извлечен 2D-срез из 3D-томограммы.",
            )
            slice_axis = axis_labels[axis_label]
            slice_count = get_slice_count(volume.shape, slice_axis)
            default_slice_index = slice_count // 2

            st.write(
                f"Размерность томограммы: `{volume.shape}` · "
                f"Срезов в выбранной плоскости: `{slice_count}`"
            )
            slice_index = st.slider(
                "Номер среза",
                min_value=0,
                max_value=slice_count - 1,
                value=default_slice_index,
                step=1,
                help="Перемещайте slider, чтобы выбрать нужный срез томограммы для обработки.",
            )
            selected_slice = extract_nifti_slice(volume, slice_axis, slice_index)
            if not np.any(selected_slice > 0):
                st.warning(
                    "Выбранный срез пустой или почти не содержит сигнала. "
                    "Попробуйте другой номер среза или плоскость."
                )
            preprocessed_mri = preprocess_mri_slice(selected_slice)
        else:
            pil_image = Image.open(uploaded_file)
            preprocessed_mri = preprocess_mri_slice(pil_image)
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Файл не читается или имеет неподдерживаемый формат: {exc}")
        return

    synthetic_ct = generate_synthetic_ct(preprocessed_mri)
    synthetic_ldct = simulate_low_dose_ct(
        synthetic_ct,
        poisson_peak=poisson_peak,
        gaussian_sigma=gaussian_sigma,
        contrast_factor=contrast_factor,
        blur_sigma=blur_sigma,
        seed=int(random_seed),
    )
    metric_values = calculate_metrics(synthetic_ct, synthetic_ldct)
    metrics_table = pd.DataFrame([metric_values]).round(4)

    col_mri, col_ct, col_ldct = st.columns(3)
    col_mri.image(
        image_to_uint8(preprocessed_mri),
        caption="Предобработанный МРТ-срез",
        use_column_width=True,
    )
    col_ct.image(
        image_to_uint8(synthetic_ct),
        caption="Синтетическое КТ",
        use_column_width=True,
    )
    col_ldct.image(
        image_to_uint8(synthetic_ldct),
        caption="Синтетическое НДКТ",
        use_column_width=True,
    )

    st.subheader("Метрики качества: синтетическое КТ vs синтетическое НДКТ")
    st.dataframe(metrics_table, use_container_width=True)

    if st.button("Сохранить результаты в outputs"):
        result_dir = save_results(preprocessed_mri, synthetic_ct, synthetic_ldct, metric_values)
        st.success(f"Результаты сохранены: {result_dir}")


if __name__ == "__main__":
    main()
