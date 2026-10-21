"""Числовой анализ распределения яркости: контрольные точки, статистика, сечения.

Результаты этого модуля требуются в отчёте: яркость в контрольных точках
видимой части сферы, а также максимальное и минимальное значения.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import LuminanceField


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
        probes: До трёх различных контрольных точек в видимой области.
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
            probes=self._probes(field),
            maximum=self._pixel_point(field, "Максимум", np.nanargmax(masked)),
            minimum=self._pixel_point(field, "Минимум", np.nanargmin(masked)),
            mean_w_m2_sr=float(values.mean()),
            pixel_count=int(values.size),
        )

    def extract_cross_sections(self, field: LuminanceField) -> tuple[CrossSection, CrossSection]:
        """Возвращает сечения через наиболее широкую видимую часть сферы.

        Args:
            field: Рассчитанное поле яркости.

        Returns:
            Сечения вдоль оси X (столбец) и вдоль оси Y (строка).
        """
        row = int(np.argmax(field.sphere_mask.sum(axis=1)))
        column = int(np.argmax(field.sphere_mask.sum(axis=0)))
        masked = np.where(field.sphere_mask, field.values_w_m2_sr, np.nan)

        return (
            CrossSection("Вдоль X", field.x_coordinates_mm, masked[:, column]),
            CrossSection("Вдоль Y", field.y_coordinates_mm, masked[row, :]),
        )

    @staticmethod
    def _probes(field: LuminanceField) -> tuple[SurfacePoint, ...]:
        """Выбирает до трёх различных пикселей видимой части сферы.

        Первый ближайший к центру видимого фрагмента, остальные — максимально
        удалённые от уже выбранных. Даже при обрезке кадра контрольные точки
        лежат внутри изображения. Если виден один-два пикселя, возвращает
        только доступные разные точки.
        """
        coordinates = np.argwhere(field.sphere_mask)
        if coordinates.size == 0:
            return ()
        center = coordinates.mean(axis=0)
        selected = [int(np.argmin(np.sum((coordinates - center) ** 2, axis=1)))]
        while len(selected) < min(3, len(coordinates)):
            distances = np.min(
                np.sum((coordinates[:, None, :] - coordinates[selected][None, :, :]) ** 2, axis=2),
                axis=1,
            )
            selected.append(int(np.argmax(distances)))
        titles = ("Центр видимой области", "Контрольная точка 2", "Контрольная точка 3")
        return tuple(
            FieldAnalyzer._pixel_point(
                field, title, np.ravel_multi_index(tuple(coordinates[index]), field.sphere_mask.shape),
            )
            for title, index in zip(titles, selected)
        )

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
