"""Числовой анализ распределения яркости: контрольные точки, статистика, сечения.

Результаты этого модуля требуются в отчёте: яркость в трёх различных точках
сферы в абсолютных величинах, а также максимальное и минимальное значения
яркости по видимой части сферы.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import intersect_sphere, project_to_screen
from .models import SceneParameters
from .physics import LuminanceCalculator, LuminanceField

# Контрольные точки задаются смещениями от проекции центра сферы на экран в
# долях видимого радиуса. Доля 0.5 гарантирует попадание на сферу при любом
# положении сферы внутри пирамиды видимости.
PROBE_OFFSETS: tuple[tuple[str, float, float], ...] = (
    ("Центр диска", 0.0, 0.0),
    ("Смещение +x", 0.5, 0.0),
    ("Смещение −y", 0.0, -0.5),
)


@dataclass(frozen=True)
class SurfacePoint:
    """Точка поверхности сферы с рассчитанной яркостью.

    Attributes:
        title: Название точки в отчёте.
        screen_mm: Координаты (x, y) пикселя на экране, мм.
        point_mm: Координаты (x, y, z) точки на сфере, мм.
        luminance_w_m2_sr: Яркость L, Вт/(м^2*ср).
    """

    title: str
    screen_mm: tuple[float, float]
    point_mm: tuple[float, float, float]
    luminance_w_m2_sr: float


@dataclass(frozen=True)
class FieldStatistics:
    """Контрольные значения и статистика яркости по видимой части сферы.

    Attributes:
        probes: Три контрольные точки.
        maximum: Точка максимальной яркости.
        minimum: Точка минимальной яркости.
        mean_w_m2_sr: Средняя яркость по пикселям сферы.
        pixel_count: Число пикселей, в которых видна сфера.
    """

    probes: tuple[SurfacePoint, ...]
    maximum: SurfacePoint
    minimum: SurfacePoint
    mean_w_m2_sr: float
    pixel_count: int


@dataclass(frozen=True)
class CrossSection:
    """Сечение распределения яркости вдоль одной оси экрана.

    Attributes:
        title: Подпись сечения для легенды графика.
        coordinates_mm: Координаты точек сечения на экране, мм.
        values_w_m2_sr: Яркость; вне сферы — NaN (разрыв линии графика).
    """

    title: str
    coordinates_mm: np.ndarray
    values_w_m2_sr: np.ndarray


class FieldAnalyzer:
    """Сбор контрольных значений, статистики и сечений распределения."""

    def __init__(self, parameters: SceneParameters, calculator: LuminanceCalculator) -> None:
        """Сохраняет параметры сцены и вычислитель для расчёта точных значений.

        Args:
            parameters: Параметры сцены.
            calculator: Вычислитель яркости той же сцены.
        """
        self._parameters = parameters
        self._calculator = calculator

    def collect_statistics(self, field: LuminanceField) -> FieldStatistics:
        """Формирует контрольные значения и статистику по видимой части сферы.

        Args:
            field: Рассчитанное поле яркости.

        Returns:
            Статистика для отчёта.

        Raises:
            ValueError: Если сфера не попала ни в один пиксель.
        """
        mask = field.sphere_mask
        values = field.values_w_m2_sr[mask]
        if values.size == 0:
            raise ValueError("Сфера не попала ни в один пиксель изображения.")

        masked = np.where(mask, field.values_w_m2_sr, np.nan)
        return FieldStatistics(
            probes=self._probes(),
            maximum=self._pixel_point(field, "Максимум", np.nanargmax(masked)),
            minimum=self._pixel_point(field, "Минимум", np.nanargmin(masked)),
            mean_w_m2_sr=float(values.mean()),
            pixel_count=int(values.size),
        )

    def extract_cross_sections(self, field: LuminanceField) -> tuple[CrossSection, CrossSection]:
        """Возвращает сечения через проекцию центра сферы вдоль осей X и Y.

        Args:
            field: Рассчитанное поле яркости.

        Returns:
            Сечения вдоль оси X (столбец) и вдоль оси Y (строка).
        """
        center_x, center_y = project_to_screen(
            self._parameters, np.asarray(self._parameters.sphere_center_mm)
        )
        row = int(np.argmin(np.abs(field.x_coordinates_mm - center_x)))
        column = int(np.argmin(np.abs(field.y_coordinates_mm - center_y)))
        masked = np.where(field.sphere_mask, field.values_w_m2_sr, np.nan)

        return (
            CrossSection("Вдоль X (y = const)", field.x_coordinates_mm, masked[:, column]),
            CrossSection("Вдоль Y (x = const)", field.y_coordinates_mm, masked[row, :]),
        )

    def _probes(self) -> tuple[SurfacePoint, ...]:
        """Рассчитывает яркость в трёх контрольных точках сферы.

        Видимый радиус сферы на экране оценивается как R * zO / (zO - zC).

        Returns:
            Контрольные точки с точными (не растровыми) значениями яркости.
        """
        parameters = self._parameters
        center_x, center_y = project_to_screen(parameters, np.asarray(parameters.sphere_center_mm))
        apparent_radius = (
            parameters.sphere_radius_mm
            * parameters.observer_z_mm
            / (parameters.observer_z_mm - parameters.sphere_z_mm)
        )

        probes = []
        for title, x_fraction, y_fraction in PROBE_OFFSETS:
            screen_x = center_x + x_fraction * apparent_radius
            screen_y = center_y + y_fraction * apparent_radius
            surface = intersect_sphere(parameters, np.array([screen_x]), np.array([screen_y]))
            luminance = self._calculator.luminance(surface.points_mm, surface.normals)
            probes.append(
                SurfacePoint(
                    title=title,
                    screen_mm=(screen_x, screen_y),
                    point_mm=tuple(float(value) for value in surface.points_mm[0]),
                    luminance_w_m2_sr=float(luminance[0]),
                )
            )
        return tuple(probes)

    @staticmethod
    def _pixel_point(field: LuminanceField, title: str, flat_index: np.intp) -> SurfacePoint:
        """Описывает пиксель растра как точку поверхности сферы.

        Args:
            field: Поле яркости.
            title: Название точки.
            flat_index: Плоский индекс пикселя в растре.

        Returns:
            Точка сферы с яркостью в этом пикселе.
        """
        row, column = np.unravel_index(flat_index, field.values_w_m2_sr.shape)
        return SurfacePoint(
            title=title,
            screen_mm=(float(field.x_coordinates_mm[row]), float(field.y_coordinates_mm[column])),
            point_mm=tuple(float(value) for value in field.surface.points_mm[row, column]),
            luminance_w_m2_sr=float(field.values_w_m2_sr[row, column]),
        )
