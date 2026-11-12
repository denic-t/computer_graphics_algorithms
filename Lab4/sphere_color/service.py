"""Полный сценарий расчёта и сравнения спектральной и RGB-сцены."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analysis import CrossSection, FieldAnalyzer, FieldStatistics
from .colorimetry import monitor_images
from .models import ParameterValidator, SceneParameters
from .physics import LuminanceCalculator, LuminanceField
from .rendering import format_report, save_images_and_report


@dataclass(frozen=True)
class ComputationResult:
    parameters: SceneParameters
    field: LuminanceField
    image_1: np.ndarray
    image_2: np.ndarray
    image_1_gray: np.ndarray
    image_2_gray: np.ndarray
    statistics: FieldStatistics
    cross_sections: tuple[CrossSection, CrossSection]
    mean_difference: float
    max_difference: int
    report: str

    @property
    def images(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.image_1, self.image_2, self.image_1_gray, self.image_2_gray


class LuminanceService:
    @staticmethod
    def compute(parameters: SceneParameters) -> ComputationResult:
        ParameterValidator.validate(parameters)
        calculator = LuminanceCalculator(parameters)
        field = calculator.compute_field()
        analyzer = FieldAnalyzer()
        statistics = analyzer.collect_statistics(field)
        images = monitor_images(field.xyz_spectral, field.xyz_rgb)
        difference = np.abs(images[0].astype(np.int16) - images[1].astype(np.int16))
        visible = difference[field.sphere_mask]
        mean_difference = float(visible.mean())
        max_difference = int(visible.max())
        return ComputationResult(
            parameters, field, *images, statistics, analyzer.extract_cross_sections(field),
            mean_difference, max_difference,
            format_report(statistics, mean_difference, max_difference),
        )

    @staticmethod
    def save(result: ComputationResult, directory: Path) -> tuple[Path, ...]:
        return save_images_and_report(directory, result.images, result.report)

    @staticmethod
    def compare_white_and_colored(parameters: SceneParameters) -> dict[str, ComputationResult]:
        """Два требуемых варианта, при одинаковой геометрии и интенсивностях."""
        return {
            "white": LuminanceService.compute(parameters.with_white_lights()),
            "colored": LuminanceService.compute(parameters),
        }

    @staticmethod
    def save_comparison(results: dict[str, ComputationResult], directory: Path) -> tuple[Path, ...]:
        files: list[Path] = []
        for name, result in results.items():
            files.extend(LuminanceService.save(result, directory / name))
        return tuple(files)
