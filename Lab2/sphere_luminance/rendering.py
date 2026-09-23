"""Представление результатов: нормировка к градациям серого и запись в файлы.

Модуль отвечает за нормировку яркости к диапазону 0-255, запись изображения
и текстового отчёта с расчётными значениями. Физических расчётов и элементов
интерфейса здесь нет.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .analysis import FieldStatistics, SurfacePoint
from .models import SceneParameters
from .physics import LuminanceField

# Максимальный уровень градаций серого 8-битного изображения.
MAXIMUM_GREY_LEVEL = 255


class LuminanceNormalizer:
    """Нормировка распределения яркости к 8-битным градациям серого."""

    @staticmethod
    def to_grey_levels(field: LuminanceField) -> np.ndarray:
        """Нормирует яркость на её максимум по видимой части сферы.

            G = round(255 * L / L_max)  в пикселях сферы,  G = 0  вне сферы.

        Args:
            field: Рассчитанное поле яркости.

        Returns:
            Матрица uint8 размера Hres x Wres.

        Raises:
            ValueError: Если максимальная яркость равна нулю (сфера не освещена).
        """
        maximum = float(field.values_w_m2_sr[field.sphere_mask].max(initial=0.0))
        if maximum <= 0.0:
            raise ValueError("Видимая часть сферы не освещена: нормировка невозможна.")

        return np.rint(field.values_w_m2_sr / maximum * MAXIMUM_GREY_LEVEL).astype(np.uint8)


class GreyscaleImageWriter:
    """Запись 8-битного полутонового изображения в графический файл."""

    @staticmethod
    def save(grey_levels: np.ndarray, file_path: Path) -> Path:
        """Сохраняет матрицу градаций серого в файл изображения.

        Args:
            grey_levels: Матрица uint8 со значениями 0-255.
            file_path: Путь к файлу; недостающие каталоги создаются.

        Returns:
            Путь к записанному файлу.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(grey_levels, mode="L").save(file_path)
        return file_path


class TextReport:
    """Текстовое представление исходных данных и расчётных значений."""

    @staticmethod
    def format(parameters: SceneParameters, statistics: FieldStatistics) -> str:
        """Формирует многострочный отчёт для панели результатов и файла.

        Args:
            parameters: Параметры сцены.
            statistics: Статистика по сфере.

        Returns:
            Текст отчёта.
        """
        lines = ["Яркость в контрольных точках, Вт/(м²·ср):"]
        lines.extend(TextReport._point_line(probe) for probe in statistics.probes)
        lines.append("")
        lines.append(TextReport._point_line(statistics.maximum))
        lines.append(TextReport._point_line(statistics.minimum))
        lines.append(
            f"  Среднее по сфере: {statistics.mean_w_m2_sr:.6e}   "
            f"(пикселей сферы: {statistics.pixel_count}, "
            f"включено источников: {len(parameters.active_lights)})"
        )
        return "\n".join(lines)

    @staticmethod
    def save(text: str, file_path: Path) -> Path:
        """Записывает отчёт в текстовый файл в кодировке UTF-8.

        Args:
            text: Текст отчёта.
            file_path: Путь к файлу.

        Returns:
            Путь к записанному файлу.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(text + "\n", encoding="utf-8")
        return file_path

    @staticmethod
    def _point_line(point: SurfacePoint) -> str:
        """Одна строка отчёта с координатами точки сферы и яркостью."""
        x, y, z = point.point_mm
        return (
            f"  {point.title:<13} P = ({x:8.2f}, {y:8.2f}, {z:8.2f}) мм:  "
            f"{point.luminance_w_m2_sr:.6e}"
        )
