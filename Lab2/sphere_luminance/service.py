"""Прикладной слой: полный цикл расчёта по одному набору параметров.

Модуль связывает проверку параметров, трассировку лучей, расчёт яркости,
нормировку и числовой анализ в одну операцию, поэтому интерфейс работает с
готовым результатом и не зависит от порядка вызова расчётных модулей.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analysis import CrossSection, FieldAnalyzer, FieldStatistics
from .models import ParameterValidator, SceneParameters
from .physics import LuminanceCalculator, LuminanceField
from .rendering import GreyscaleImageWriter, LuminanceNormalizer, TextReport


@dataclass(frozen=True)
class ComputationResult:
    """Полный результат расчёта для одного набора параметров сцены.

    Attributes:
        parameters: Исходные данные, по которым выполнен расчёт.
        field: Поле яркости в абсолютных значениях, Вт/(м^2*ср).
        grey_levels: Нормированное изображение 0-255.
        statistics: Контрольные точки и статистика по сфере.
        cross_sections: Сечения через проекцию центра сферы.
        report: Текстовый отчёт с расчётными значениями.
    """

    parameters: SceneParameters
    field: LuminanceField
    grey_levels: np.ndarray
    statistics: FieldStatistics
    cross_sections: tuple[CrossSection, CrossSection]
    report: str


class LuminanceService:
    """Сценарий расчёта распределения яркости на сфере и сохранения результата."""

    @staticmethod
    def compute(parameters: SceneParameters) -> ComputationResult:
        """Выполняет расчёт целиком.

        Последовательность шагов:
            1. проверка параметров на соответствие ограничениям задачи;
            2. трассировка лучей из наблюдателя через пиксели экрана;
            3. расчёт яркости в видимых точках сферы по модели Блинн-Фонга;
            4. нормировка к градациям серого 0-255 на максимум яркости;
            5. сбор контрольных значений, статистики и сечений.

        Args:
            parameters: Набор исходных данных сцены.

        Returns:
            Результат расчёта, пригодный для визуализации и сохранения.

        Raises:
            ParameterValidationError: Если параметры нарушают ограничения.
            ValueError: Если сфера не видна или не освещена.
        """
        ParameterValidator.validate(parameters)

        calculator = LuminanceCalculator(parameters)
        field = calculator.compute_field()
        analyzer = FieldAnalyzer(parameters, calculator)
        statistics = analyzer.collect_statistics(field)

        return ComputationResult(
            parameters=parameters,
            field=field,
            grey_levels=LuminanceNormalizer.to_grey_levels(field),
            statistics=statistics,
            cross_sections=analyzer.extract_cross_sections(field),
            report=TextReport.format(parameters, statistics),
        )

    @staticmethod
    def save(result: ComputationResult, image_path: Path) -> tuple[Path, Path]:
        """Записывает изображение и рядом — текстовый файл с расчётными значениями.

        Args:
            result: Результат расчёта.
            image_path: Путь к файлу изображения; отчёт получает то же имя с
                суффиксом «_values.txt».

        Returns:
            Пути к файлу изображения и файлу отчёта.
        """
        report_path = image_path.with_name(f"{image_path.stem}_values.txt")
        return (
            GreyscaleImageWriter.save(result.grey_levels, image_path),
            TextReport.save(result.report, report_path),
        )
