"""Таблицы CIE 1931, стандартный D65 и преобразования цвета ЛР3."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .models import WAVELENGTHS_NM

STEP_NM = 10.0

# Матрица CIE RGB 1931 → XYZ из задания. В приведённой там обратной
# матрице опущен множитель 0.17697, поэтому обратную вычисляем точно.
RGB_TO_XYZ = np.array([
    [0.49000, 0.31000, 0.20000],
    [0.17697, 0.81240, 0.01063],
    [0.00000, 0.01000, 0.99000],
]) / 0.17697
XYZ_TO_RGB = np.linalg.inv(RGB_TO_XYZ)

# Линейный sRGB (D65) нужен только для вывода на обычный монитор.
XYZ_TO_LINEAR_SRGB = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
])
SRGB_LUMINANCE = np.array([0.2126, 0.7152, 0.0722])


@dataclass(frozen=True)
class ColorTables:
    wavelengths_nm: np.ndarray
    xyz_matching: np.ndarray
    d65: np.ndarray


def _read_table(path: Path, headers: tuple[str, ...]) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [[float(row[key]) for key in headers] for row in reader]
    result = np.asarray(rows)
    if result.shape != (len(WAVELENGTHS_NM), len(headers)):
        raise ValueError(f"Неполная спектральная таблица: {path}")
    if not np.array_equal(result[:, 0], WAVELENGTHS_NM) or not np.all(np.isfinite(result)):
        raise ValueError(f"Неверная сетка длин волн или некорректные числа: {path}")
    return result


@lru_cache(maxsize=1)
def load_tables() -> ColorTables:
    """Читает автономные CSV, полученные из приложенного XLS и CIE D65."""
    directory = Path(__file__).resolve().parent / "data"
    cie = _read_table(directory / "cie_xyz_1931_10nm.csv", ("wavelength_nm", "x_bar", "y_bar", "z_bar"))
    d65 = _read_table(directory / "d65_10nm.csv", ("wavelength_nm", "relative_power"))
    return ColorTables(cie[:, 0], cie[:, 1:], d65[:, 1])


def spectrum_to_xyz(spectrum: np.ndarray) -> np.ndarray:
    """Интегрирует по длине волны; последний размер входа равен 41."""
    matching = load_tables().xyz_matching
    values = np.asarray(spectrum, dtype=float)
    if values.shape[-1] != len(WAVELENGTHS_NM):
        raise ValueError("Спектр должен содержать 41 значение.")
    return np.trapezoid(values[..., :, None] * matching, dx=STEP_NM, axis=-2)


def xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    """Абсолютные тристимульные значения CIE RGB 1931, без отсечения знака."""
    return np.asarray(xyz) @ XYZ_TO_RGB.T


def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(rgb) @ RGB_TO_XYZ.T


def reflectance_to_rgb(reflectance: np.ndarray) -> np.ndarray:
    """Коэффициенты RGB = RGB(D65 × Kd) / RGB(D65), как в задании."""
    d65 = load_tables().d65
    reference = xyz_to_rgb(spectrum_to_xyz(d65))
    reflected = xyz_to_rgb(spectrum_to_xyz(d65 * reflectance))
    if np.any(np.abs(reference) < 1e-12):
        raise ValueError("Не удалось вычислить коэффициенты отражения относительно D65.")
    return reflected / reference


def encode_srgb(linear: np.ndarray) -> np.ndarray:
    """IEC sRGB transfer function on values already limited to [0, 1]."""
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)


def monitor_images(
    xyz_spectral: np.ndarray, xyz_rgb: np.ndarray, *, reference_y: float | None = None,
) -> tuple[np.ndarray, ...]:
    """Общая нормировка по Y; фиксированный reference_y нужен только для просмотра."""
    maximum_y = (
        reference_y if reference_y is not None else
        max(float(np.max(xyz_spectral[..., 1])), float(np.max(xyz_rgb[..., 1])))
    )
    if not np.isfinite(maximum_y) or maximum_y <= 0:
        raise ValueError("Нужен положительный уровень экспозиции: видимая часть сферы не освещена.")
    images = []
    for xyz in (xyz_spectral, xyz_rgb):
        linear = np.clip((xyz / maximum_y) @ XYZ_TO_LINEAR_SRGB.T, 0, 1)
        color = np.rint(255 * encode_srgb(linear)).astype(np.uint8)
        grey_linear = np.clip(linear @ SRGB_LUMINANCE, 0, 1)
        grey = np.rint(255 * encode_srgb(grey_linear)).astype(np.uint8)
        images.append((color, grey))
    return images[0][0], images[1][0], images[0][1], images[1][1]
