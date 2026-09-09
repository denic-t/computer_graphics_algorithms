"""Расчёт освещённости плоскости z = 0 точечным ламбертовским источником.

Модуль содержит только физическую модель и построение расчётной сетки.
Он ничего не знает ни о визуализации, ни о нормировке изображения.

Расчётная модель
----------------
Источник L(xL, yL, zL) — точечный, с ламбертовской диаграммой излучения,
ось диаграммы направлена вертикально вниз (вдоль -Z), на приёмную плоскость.

    (1) Сила излучения в направлении theta:      I(theta) = I0 * cos(theta)
    (2) Расстояние от источника до точки P(x, y, 0):
            r^2 = (x - xL)^2 + (y - yL)^2 + zL^2
    (3) Косинус угла между осью диаграммы (-Z) и направлением L -> P:
            cos(theta) = zL / r
    (4) Косинус угла падения на площадку с нормалью n = (0, 0, 1)
        совпадает с (3), так как плоскость горизонтальна:
            cos(alpha) = zL / r
    (5) Освещённость (закон обратных квадратов и закон косинуса):
            E = I(theta) * cos(alpha) / r^2

Подстановка (1), (3) и (4) в (5) даёт итоговую рабочую формулу:

    (6) E(x, y) = I0 * zL^2 / r^4
               = I0 * zL^2 / ((x - xL)^2 + (y - yL)^2 + zL^2)^2

Все геометрические параметры задаются в миллиметрах, но расчёт ведётся в
метрах, поэтому освещённость получается в Вт/м^2 при силе излучения в Вт/ср.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import SceneParameters

# Число миллиметров в метре: параметры сцены задаются в мм, а освещённость
# приводится к системным единицам Вт/м^2.
MILLIMETRES_PER_METRE = 1000.0


@dataclass(frozen=True)
class IlluminanceField:
    """Результат расчёта распределения освещённости по области изображения.

    Attributes:
        values_w_m2: Матрица освещённости, Вт/м^2. Строки соответствуют оси X
            (первая строка — максимальное x), столбцы — оси Y (первый столбец
            — минимальное y).
        x_coordinates_mm: Координаты x центров строк, мм (по убыванию).
        y_coordinates_mm: Координаты y центров столбцов, мм (по возрастанию).
        circle_mask: Булева маска пикселей, попавших внутрь расчётного круга.
    """

    values_w_m2: np.ndarray
    x_coordinates_mm: np.ndarray
    y_coordinates_mm: np.ndarray
    circle_mask: np.ndarray

    @property
    def masked_values_w_m2(self) -> np.ndarray:
        """Освещённость, обнулённая вне расчётного круга, Вт/м^2."""
        return np.where(self.circle_mask, self.values_w_m2, 0.0)


class IlluminanceCalculator:
    """Вычислитель освещённости для одного набора параметров сцены."""

    def __init__(self, parameters: SceneParameters) -> None:
        """Сохраняет параметры сцены для последующих расчётов.

        Args:
            parameters: Проверенный набор исходных данных сцены.
        """
        self._parameters = parameters

    def illuminance_at(self, x_mm: float, y_mm: float) -> float:
        """Вычисляет освещённость в одной точке плоскости по формуле (6).

        Используется для получения точных вещественных значений в контрольных
        точках, независимо от дискретизации изображения.

        Args:
            x_mm: Координата x точки на плоскости z = 0, мм.
            y_mm: Координата y точки на плоскости z = 0, мм.

        Returns:
            Освещённость в точке, Вт/м^2.
        """
        return float(
            self._illuminance(np.asarray(x_mm, dtype=float), np.asarray(y_mm, dtype=float))
        )

    def compute_field(self) -> IlluminanceField:
        """Рассчитывает распределение освещённости по всей области изображения.

        Расчёт выполняется в центрах пикселей: значение пикселя отвечает
        освещённости в геометрическом центре соответствующей ячейки растра.
        Дополнительно строится маска круга

            (x - xc)^2 + (y - yc)^2 <= R^2,

        внутри которого распределение подлежит визуализации.

        Returns:
            Поле освещённости вместе с координатами сетки и маской круга.
        """
        x_coordinates_mm, y_coordinates_mm = self._pixel_centres_mm()

        # Сетка строится через broadcasting: столбец x-координат на строку
        # y-координат, что избавляет от поэлементных циклов по растру.
        x_grid_mm = x_coordinates_mm[:, np.newaxis]
        y_grid_mm = y_coordinates_mm[np.newaxis, :]

        values_w_m2 = self._illuminance(x_grid_mm, y_grid_mm)
        circle_mask = self._circle_mask(x_grid_mm, y_grid_mm)

        return IlluminanceField(
            values_w_m2=values_w_m2,
            x_coordinates_mm=x_coordinates_mm,
            y_coordinates_mm=y_coordinates_mm,
            circle_mask=circle_mask,
        )

    def _pixel_centres_mm(self) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает координаты центров пикселей расчётной сетки.

        Центр пикселя с индексом i смещён от границы области на половину
        размера пикселя:

            x_i = H/2 - (i + 0.5) * H / Hres,   i = 0 .. Hres - 1
            y_j = -W/2 + (j + 0.5) * W / Wres,  j = 0 .. Wres - 1

        Координаты x идут по убыванию, чтобы первая строка матрицы
        соответствовала верхней строке изображения.

        Returns:
            Кортеж (координаты x строк, координаты y столбцов) в мм.
        """
        parameters = self._parameters
        x_max_mm = parameters.height_mm / 2.0
        y_min_mm = -parameters.width_mm / 2.0

        row_indices = np.arange(parameters.height_px, dtype=float)
        column_indices = np.arange(parameters.width_px, dtype=float)

        x_coordinates_mm = x_max_mm - (row_indices + 0.5) * parameters.pixel_height_mm
        y_coordinates_mm = y_min_mm + (column_indices + 0.5) * parameters.pixel_width_mm

        return x_coordinates_mm, y_coordinates_mm

    def _illuminance(self, x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
        """Применяет рабочую формулу (6) к массивам координат.

        Args:
            x_mm: Координаты x точек плоскости, мм.
            y_mm: Координаты y точек плоскости, мм.

        Returns:
            Массив освещённости той же формы (после broadcasting), Вт/м^2.
        """
        parameters = self._parameters

        # Перевод в метры: E [Вт/м^2] требует расстояний в метрах.
        offset_x_m = (x_mm - parameters.light_x_mm) / MILLIMETRES_PER_METRE
        offset_y_m = (y_mm - parameters.light_y_mm) / MILLIMETRES_PER_METRE
        light_height_m = parameters.light_z_mm / MILLIMETRES_PER_METRE

        # r^2 = (x - xL)^2 + (y - yL)^2 + zL^2 — формула (2).
        squared_distance_m2 = offset_x_m**2 + offset_y_m**2 + light_height_m**2

        # E = I0 * zL^2 / r^4 — формула (6); r^4 записан как (r^2)^2.
        return (
            parameters.axial_intensity_w_sr
            * light_height_m**2
            / squared_distance_m2**2
        )

    def _circle_mask(self, x_grid_mm: np.ndarray, y_grid_mm: np.ndarray) -> np.ndarray:
        """Строит маску принадлежности пикселей расчётному кругу.

        Критерий принадлежности:

            (x - xc)^2 + (y - yc)^2 <= R^2

        Args:
            x_grid_mm: Столбец координат x центров пикселей, мм.
            y_grid_mm: Строка координат y центров пикселей, мм.

        Returns:
            Булева матрица размера Hres x Wres.
        """
        parameters = self._parameters
        offset_x_mm = x_grid_mm - parameters.circle_center_x_mm
        offset_y_mm = y_grid_mm - parameters.circle_center_y_mm
        squared_radius_mm2 = parameters.circle_radius_mm**2

        return offset_x_mm**2 + offset_y_mm**2 <= squared_radius_mm2
