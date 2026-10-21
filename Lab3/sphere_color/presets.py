"""Готовые спектры и простая модель спектральной полосы для редактора."""

from __future__ import annotations

import numpy as np

from .colorimetry import load_tables
from .models import Spectrum, WAVELENGTHS_NM


def white_d65() -> Spectrum:
    """Стандартный белый источник D65; I0 задаёт общую мощность отдельно."""
    values = load_tables().d65
    return Spectrum.from_values(values / values.max())


def neutral_reflectance() -> Spectrum:
    """Нейтральная поверхность с постоянным коэффициентом 0,7."""
    return Spectrum.from_values([0.7] * len(WAVELENGTHS_NM))


def gaussian_band(peak_nm: int, width_nm: int, *, reflectance: bool = False) -> Spectrum:
    """Наглядный модельный спектр с гауссовой полосой шириной σ.

    Это удобный учебный шаблон, а не измеренный спектр конкретной лампы.
    Для источника слабый фон не даёт нулей за пределами полосы; для
    отражения значения всегда остаются в интервале [0, 1].
    """
    if not 380 <= peak_nm <= 780 or not 10 <= width_nm <= 150:
        raise ValueError("Пик должен быть в диапазоне 380–780 нм, ширина σ — 10–150 нм.")
    wavelengths = np.asarray(WAVELENGTHS_NM)
    band = np.exp(-0.5 * ((wavelengths - peak_nm) / width_nm) ** 2)
    values = 0.1 + 0.8 * band if reflectance else 0.02 + 0.98 * band
    return Spectrum.from_values(values)
