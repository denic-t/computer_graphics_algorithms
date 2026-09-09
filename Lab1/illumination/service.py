"""Прикладной слой: полный цикл расчёта по одному набору параметров.

Модуль связывает проверку параметров, физический расчёт, нормировку и
числовой анализ в одну операцию. Благодаря этому интерфейс пользователя
работает с готовым результатом и не зависит от порядка вызова расчётных
модулей.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analysis import CrossSection, FieldAnalyzer, FieldStatistics
from .models import ParameterValidator, SceneParameters
from .physics import IlluminanceCalculator, IlluminanceField
from .rendering import (
    GreyscaleImageWriter,
    IlluminanceNormalizer,
    SourceMarkerPainter,
)


@dataclass(frozen=True)
class ComputationResult:
    """Полный результат расчёта для одного набора параметров сцены.

    Attributes:
        parameters: Исходные данные, по которым выполнен расчёт.
        field: Поле освещённости в вещественных значениях, Вт/м^2.
        grey_levels: Нормированное изображение распределения, 0-255, с
            нанесённой меткой проекции источника света.
        statistics: Контрольные значения и статистика по кругу.
        cross_sections: Сечения распределения вдоль осей X и Y.
    """

    parameters: SceneParameters
    field: IlluminanceField
    grey_levels: np.ndarray
    statistics: FieldStatistics
    cross_sections: tuple[CrossSection, CrossSection]


class IlluminanceService:
    """Сценарий расчёта распределения освещённости и сохранения результата."""

    @staticmethod
    def compute(parameters: SceneParameters) -> ComputationResult:
        """Выполняет расчёт распределения освещённости целиком.

        Последовательность шагов:
            1. проверка параметров на соответствие пределам задачи;
            2. расчёт поля освещённости по формуле E = I0 * zL^2 / r^4;
            3. нормировка значений внутри круга к градациям серого 0-255;
            4. нанесение метки проекции источника света на изображение;
            5. сбор контрольных значений, статистики и сечений.

        Args:
            parameters: Набор исходных данных сцены.

        Returns:
            Результат расчёта, пригодный для визуализации и сохранения.

        Raises:
            ParameterValidationError: Если параметры нарушают ограничения.
            ValueError: Если внутрь круга не попал ни один пиксель растра.
        """
        ParameterValidator.validate(parameters)

        calculator = IlluminanceCalculator(parameters)
        field = calculator.compute_field()

        analyzer = FieldAnalyzer(parameters, calculator)
        grey_levels = IlluminanceNormalizer.to_grey_levels(field)

        return ComputationResult(
            parameters=parameters,
            field=field,
            grey_levels=SourceMarkerPainter.paint(grey_levels, field, parameters),
            statistics=analyzer.collect_statistics(field),
            cross_sections=analyzer.extract_cross_sections(field),
        )

    @staticmethod
    def save_image(result: ComputationResult, file_path: Path) -> Path:
        """Записывает нормированное изображение распределения в файл.

        Args:
            result: Результат расчёта.
            file_path: Путь к создаваемому файлу изображения.

        Returns:
            Путь к записанному файлу.
        """
        return GreyscaleImageWriter.save(result.grey_levels, file_path)
