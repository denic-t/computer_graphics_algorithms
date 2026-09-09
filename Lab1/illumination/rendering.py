"""Преобразование поля освещённости в растровое изображение и запись в файл.

Модуль отвечает за нормировку вещественных значений освещённости к диапазону
градаций серого 0-255, за нанесение служебной разметки и за сохранение
результата на диск. Ни физических расчётов, ни элементов интерфейса здесь нет.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .models import SceneParameters
from .physics import IlluminanceField

# Максимальный уровень градаций серого 8-битного изображения.
MAXIMUM_GREY_LEVEL = 255

# Нейтральный светло-серый уровень метки источника света: различим и на
# тёмном фоне вне круга, и на светлом пятне вблизи проекции источника.
SOURCE_MARKER_GREY_LEVEL = 200

# Половина длины луча крестообразной метки. Задаётся долей от меньшей стороны
# растра, чтобы метка сохраняла видимый размер при любом разрешении.
SOURCE_MARKER_SIZE_FRACTION = 0.025
SOURCE_MARKER_MINIMUM_HALF_LENGTH_PX = 4


class IlluminanceNormalizer:
    """Нормировка распределения освещённости к 8-битным градациям серого."""

    @staticmethod
    def to_grey_levels(field: IlluminanceField) -> np.ndarray:
        """Нормирует освещённость на её максимум внутри расчётного круга.

        Применяемое преобразование:

            G(x, y) = round(255 * E(x, y) / E_max),  если точка внутри круга,
            G(x, y) = 0,                             иначе,

        где E_max — максимальная освещённость среди пикселей круга. Точки вне
        круга не рассчитываются по условию задачи и выводятся чёрным.

        Args:
            field: Рассчитанное поле освещённости.

        Returns:
            Матрица uint8 размера Hres x Wres со значениями 0-255.

        Raises:
            ValueError: Если внутрь круга не попал ни один пиксель растра.
        """
        values_inside_circle = field.values_w_m2[field.circle_mask]
        if values_inside_circle.size == 0:
            raise ValueError(
                "Внутрь круга не попал ни один пиксель: нормировка невозможна."
            )

        maximum_w_m2 = float(values_inside_circle.max())
        normalized = field.masked_values_w_m2 / maximum_w_m2

        return np.rint(normalized * MAXIMUM_GREY_LEVEL).astype(np.uint8)


class SourceMarkerPainter:
    """Нанесение метки проекции источника света на растр изображения."""

    @staticmethod
    def paint(
        grey_levels: np.ndarray,
        field: IlluminanceField,
        parameters: SceneParameters,
    ) -> np.ndarray:
        """Наносит крестообразную метку в точке проекции источника на плоскость.

        Проекция источника L(xL, yL, zL) на плоскость z = 0 — точка (xL, yL);
        именно в ней освещённость максимальна, поэтому метка показывает
        положение светового пятна относительно расчётной области.

        Метка наносится поверх нормированного изображения и не влияет ни на
        рассчитанные значения освещённости, ни на статистику по кругу.

        Args:
            grey_levels: Нормированное изображение 0-255.
            field: Поле освещённости с координатами центров пикселей.
            parameters: Параметры сцены с координатами источника света.

        Returns:
            Копия изображения с нанесённой меткой. Если проекция источника
            выходит за границы области изображения, возвращается копия без
            изменений.
        """
        marked = grey_levels.copy()

        x_minimum_mm, x_maximum_mm = parameters.x_bounds_mm
        y_minimum_mm, y_maximum_mm = parameters.y_bounds_mm
        inside_area = (
            x_minimum_mm <= parameters.light_x_mm <= x_maximum_mm
            and y_minimum_mm <= parameters.light_y_mm <= y_maximum_mm
        )
        if not inside_area:
            return marked

        row = int(np.argmin(np.abs(field.x_coordinates_mm - parameters.light_x_mm)))
        column = int(np.argmin(np.abs(field.y_coordinates_mm - parameters.light_y_mm)))

        row_count, column_count = marked.shape
        half_length = max(
            SOURCE_MARKER_MINIMUM_HALF_LENGTH_PX,
            int(min(row_count, column_count) * SOURCE_MARKER_SIZE_FRACTION),
        )

        # Горизонтальный и вертикальный лучи креста; границы обрезаются, если
        # проекция источника лежит у самого края растра.
        row_slice = slice(max(row - half_length, 0), min(row + half_length + 1, row_count))
        column_slice = slice(
            max(column - half_length, 0), min(column + half_length + 1, column_count)
        )

        marked[row, column_slice] = SOURCE_MARKER_GREY_LEVEL
        marked[row_slice, column] = SOURCE_MARKER_GREY_LEVEL

        return marked


class GreyscaleImageWriter:
    """Запись 8-битного полутонового изображения в графический файл."""

    @staticmethod
    def save(grey_levels: np.ndarray, file_path: Path) -> Path:
        """Сохраняет матрицу градаций серого в файл изображения.

        Args:
            grey_levels: Матрица uint8 со значениями 0-255.
            file_path: Путь к создаваемому файлу; недостающие каталоги
                создаются автоматически.

        Returns:
            Путь к записанному файлу.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(grey_levels, mode="L").save(file_path)

        return file_path
