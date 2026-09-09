"""Числовой анализ рассчитанного распределения освещённости.

Модуль отвечает за величины, которые требуется привести в отчёте:
контрольные значения освещённости в пяти точках, экстремальные и среднее
значения в пределах круга, а также сечения распределения, проходящие через
центр расчётной области.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import SceneParameters
from .physics import IlluminanceCalculator, IlluminanceField


@dataclass(frozen=True)
class ProbePoint:
    """Контрольная точка с точным (не растровым) значением освещённости.

    Attributes:
        title: Название точки для вывода в интерфейсе и отчёте.
        x_mm: Координата x точки, мм.
        y_mm: Координата y точки, мм.
        illuminance_w_m2: Освещённость в точке, Вт/м^2.
    """

    title: str
    x_mm: float
    y_mm: float
    illuminance_w_m2: float


@dataclass(frozen=True)
class CrossSection:
    """Сечение распределения освещённости вдоль одной координатной оси.

    Attributes:
        title: Название сечения для легенды графика.
        axis_label: Подпись оси абсцисс графика.
        coordinates_mm: Координаты точек сечения вдоль оси, мм.
        values_w_m2: Освещённость в точках сечения, Вт/м^2.
    """

    title: str
    axis_label: str
    coordinates_mm: np.ndarray
    values_w_m2: np.ndarray


@dataclass(frozen=True)
class FieldStatistics:
    """Сводные характеристики распределения освещённости внутри круга.

    Attributes:
        probes: Контрольные точки: центр круга и пересечения окружности с
            направлениями осей X и Y.
        maximum_w_m2: Максимальная освещённость в круге, Вт/м^2.
        minimum_w_m2: Минимальная освещённость в круге, Вт/м^2.
        mean_w_m2: Средняя освещённость в круге, Вт/м^2.
        pixel_count: Число пикселей растра, попавших внутрь круга.
    """

    probes: tuple[ProbePoint, ...]
    maximum_w_m2: float
    minimum_w_m2: float
    mean_w_m2: float
    pixel_count: int


class FieldAnalyzer:
    """Извлекает контрольные значения и сечения из поля освещённости."""

    def __init__(
        self, parameters: SceneParameters, calculator: IlluminanceCalculator
    ) -> None:
        """Связывает анализатор с параметрами сцены и вычислителем.

        Args:
            parameters: Исходные данные сцены.
            calculator: Вычислитель, используемый для точных значений в
                контрольных точках (в обход растровой сетки).
        """
        self._parameters = parameters
        self._calculator = calculator

    def collect_statistics(self, field: IlluminanceField) -> FieldStatistics:
        """Собирает контрольные значения и статистику по расчётному кругу.

        Экстремумы и среднее берутся по пикселям растра, попавшим внутрь
        круга, — именно эти значения формируют выводимое изображение.
        Контрольные точки рассчитываются аналитически по формуле освещённости,
        поэтому не зависят от разрешения растра.

        Args:
            field: Рассчитанное поле освещённости.

        Returns:
            Сводные характеристики распределения.

        Raises:
            ValueError: Если внутрь круга не попал ни один пиксель растра
                (радиус меньше половины размера пикселя).
        """
        values_inside_circle = field.values_w_m2[field.circle_mask]
        if values_inside_circle.size == 0:
            raise ValueError(
                "Внутрь круга не попал ни один пиксель: увеличьте радиус R "
                "или разрешение изображения."
            )

        return FieldStatistics(
            probes=self._collect_probes(),
            maximum_w_m2=float(values_inside_circle.max()),
            minimum_w_m2=float(values_inside_circle.min()),
            mean_w_m2=float(values_inside_circle.mean()),
            pixel_count=int(values_inside_circle.size),
        )

    def extract_cross_sections(
        self, field: IlluminanceField
    ) -> tuple[CrossSection, CrossSection]:
        """Строит два сечения распределения через центр расчётной области.

        Сечения выбираются как ближайшие к центру круга строка и столбец
        растра; учитываются только точки внутри круга, где распределение
        подлежит визуализации.

        Args:
            field: Рассчитанное поле освещённости.

        Returns:
            Кортеж (сечение вдоль оси X, сечение вдоль оси Y).
        """
        row_index = int(
            np.argmin(
                np.abs(field.x_coordinates_mm - self._parameters.circle_center_x_mm)
            )
        )
        column_index = int(
            np.argmin(
                np.abs(field.y_coordinates_mm - self._parameters.circle_center_y_mm)
            )
        )

        section_along_x = self._build_section(
            title="Сечение вдоль оси X",
            axis_label="x, мм",
            coordinates_mm=field.x_coordinates_mm,
            values_w_m2=field.values_w_m2[:, column_index],
            inside_circle=field.circle_mask[:, column_index],
        )
        section_along_y = self._build_section(
            title="Сечение вдоль оси Y",
            axis_label="y, мм",
            coordinates_mm=field.y_coordinates_mm,
            values_w_m2=field.values_w_m2[row_index, :],
            inside_circle=field.circle_mask[row_index, :],
        )

        return section_along_x, section_along_y

    def _collect_probes(self) -> tuple[ProbePoint, ...]:
        """Вычисляет освещённость в пяти контрольных точках круга.

        Контрольными точками служат центр круга и четыре точки окружности,
        смещённые от центра на радиус R вдоль направлений осей X и Y:

            (xc, yc), (xc +- R, yc), (xc, yc +- R)

        При центре круга в начале координат эти точки совпадают с
        пересечениями окружности с осями X и Y.

        Returns:
            Кортеж из пяти контрольных точек с рассчитанными значениями.
        """
        parameters = self._parameters
        center_x_mm = parameters.circle_center_x_mm
        center_y_mm = parameters.circle_center_y_mm
        radius_mm = parameters.circle_radius_mm

        layout = (
            ("Центр круга", center_x_mm, center_y_mm),
            ("Окружность, +X", center_x_mm + radius_mm, center_y_mm),
            ("Окружность, -X", center_x_mm - radius_mm, center_y_mm),
            ("Окружность, +Y", center_x_mm, center_y_mm + radius_mm),
            ("Окружность, -Y", center_x_mm, center_y_mm - radius_mm),
        )

        return tuple(
            ProbePoint(
                title=title,
                x_mm=x_mm,
                y_mm=y_mm,
                illuminance_w_m2=self._calculator.illuminance_at(x_mm, y_mm),
            )
            for title, x_mm, y_mm in layout
        )

    @staticmethod
    def _build_section(
        title: str,
        axis_label: str,
        coordinates_mm: np.ndarray,
        values_w_m2: np.ndarray,
        inside_circle: np.ndarray,
    ) -> CrossSection:
        """Отбирает точки сечения, лежащие внутри расчётного круга.

        Args:
            title: Название сечения.
            axis_label: Подпись оси абсцисс.
            coordinates_mm: Координаты всех точек линии сечения, мм.
            values_w_m2: Освещённость во всех точках линии, Вт/м^2.
            inside_circle: Маска принадлежности точек линии кругу.

        Returns:
            Сечение, содержащее только точки внутри круга.
        """
        order = np.argsort(coordinates_mm)
        sorted_mask = inside_circle[order]

        return CrossSection(
            title=title,
            axis_label=axis_label,
            coordinates_mm=coordinates_mm[order][sorted_mask],
            values_w_m2=values_w_m2[order][sorted_mask],
        )
