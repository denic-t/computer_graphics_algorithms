"""Представление и сохранение цветных, полутоновых изображений и отчёта."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .analysis import FieldStatistics, SurfacePoint


def format_report(statistics: FieldStatistics, mean_difference: float, max_difference: int) -> str:
    """Абсолютная интегральная яркость и сравнение двух методов."""
    def point_line(point: SurfacePoint) -> str:
        coords = ", ".join(f"{value:.2f}" for value in point.point_mm)
        return f"{point.title}: P = ({coords}) мм; L = {point.luminance_w_m2_sr:.8e} Вт/(м²·ср)"

    lines = [
        "ЛР3 — цветная диффузная сфера",
        "Интегральная абсолютная радианс видимой поверхности:",
        *(point_line(point) for point in statistics.probes),
        point_line(statistics.maximum),
        point_line(statistics.minimum),
        f"Среднее: {statistics.mean_w_m2_sr:.8e} Вт/(м²·ср)",
        f"Число пикселей сферы: {statistics.pixel_count}",
        "",
        "Сравнение image_1 и image_2 в мониторных единицах 0–255:",
        f"Среднее абсолютное отличие каналов на сфере: {mean_difference:.4f}",
        f"Максимальное отличие одного канала на сфере: {max_difference}",
        "Одинаковая нормировка по максимальному Y применена к обоим изображениям.",
    ]
    return "\n".join(lines)


def save_images_and_report(
    directory: Path,
    images: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    report: str,
) -> tuple[Path, ...]:
    """Сохраняет image_1/2, две полутоновые версии и текстовые значения."""
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for name, pixels in zip(("image_1", "image_2", "image_1_gray", "image_2_gray"), images):
        path = directory / f"{name}.png"
        Image.fromarray(pixels).save(path)
        files.append(path)
    report_path = directory / "values.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    files.append(report_path)
    return tuple(files)
