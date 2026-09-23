"""Геометрия наблюдения: лучи зрения через пиксели экрана и их пересечение со сферой.

Для каждого пикселя строится луч из наблюдателя O через центр пикселя на
экране S (z = 0). Первая точка пересечения луча со сферой — это точка P,
которую наблюдатель видит в данном пикселе.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import SceneParameters


@dataclass(frozen=True)
class SurfaceSample:
    """Видимые точки сферы для набора лучей зрения.

    Все массивы имеют одинаковую форму по первым осям: форма набора лучей
    (например, Hres x Wres для всего растра).

    Attributes:
        hit_mask: True, если луч пересекает сферу.
        points_mm: Координаты точек P на сфере, мм; для лучей без
            пересечения — NaN. Последняя ось — (x, y, z).
        normals: Единичные внешние нормали N = (P - C) / R; NaN вне сферы.
    """

    hit_mask: np.ndarray
    points_mm: np.ndarray
    normals: np.ndarray


def pixel_centres_mm(parameters: SceneParameters) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает координаты центров пикселей экрана.

    Строка 0 соответствует верхнему краю x = +H/2, столбец 0 — левому краю
    y = -W/2 (как в ЛР1).

    Args:
        parameters: Параметры сцены.

    Returns:
        Кортеж (x-координаты строк, y-координаты столбцов), мм.
    """
    x_min_mm, x_max_mm = parameters.x_bounds_mm
    y_min_mm, _ = parameters.y_bounds_mm

    row_indices = np.arange(parameters.height_px)
    column_indices = np.arange(parameters.width_px)

    x_coordinates_mm = x_max_mm - (row_indices + 0.5) * parameters.pixel_height_mm
    y_coordinates_mm = y_min_mm + (column_indices + 0.5) * parameters.pixel_width_mm
    return x_coordinates_mm, y_coordinates_mm


def intersect_sphere(
    parameters: SceneParameters, screen_x_mm: np.ndarray, screen_y_mm: np.ndarray
) -> SurfaceSample:
    """Находит ближайшие к наблюдателю точки пересечения лучей со сферой.

    Луч: X(t) = O + t * d, |d| = 1, d = (S - O) / |S - O|. Подстановка в
    уравнение сферы |X - C|^2 = R^2 даёт квадратное уравнение

        t^2 + 2 * b * t + c = 0,  b = d . (O - C),  c = |O - C|^2 - R^2,

    с дискриминантом D = b^2 - c. При D >= 0 ближайший корень t = -b - sqrt(D).
    Наблюдатель находится вне сферы (c > 0), поэтому оба корня одного знака, и
    t > 0 означает, что сфера лежит перед наблюдателем.

    Args:
        parameters: Параметры сцены.
        screen_x_mm: x-координаты точек экрана, мм (любая форма).
        screen_y_mm: y-координаты точек экрана, мм (та же форма).

    Returns:
        Видимые точки сферы и нормали в них.
    """
    observer = np.asarray(parameters.observer_mm)
    center = np.asarray(parameters.sphere_center_mm)
    radius = parameters.sphere_radius_mm

    screen_points = np.stack(
        np.broadcast_arrays(screen_x_mm, screen_y_mm, np.zeros_like(screen_x_mm)), axis=-1
    ).astype(float)
    directions = screen_points - observer
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    offset = observer - center
    half_b = directions @ offset
    c = offset @ offset - radius**2
    discriminant = half_b**2 - c

    hit_mask = discriminant >= 0.0
    distances = np.where(hit_mask, -half_b - np.sqrt(np.where(hit_mask, discriminant, 0.0)), np.nan)
    hit_mask &= distances > 0.0

    points_mm = observer + distances[..., np.newaxis] * directions
    points_mm[~hit_mask] = np.nan
    normals = (points_mm - center) / radius

    return SurfaceSample(hit_mask=hit_mask, points_mm=points_mm, normals=normals)


def project_to_screen(parameters: SceneParameters, point_mm: np.ndarray) -> tuple[float, float]:
    """Центральная проекция точки пространства на экран из наблюдателя.

    Точка экрана лежит на прямой O-P при z = 0: S = O + (P - O) * zO / (zO - zP).

    Args:
        parameters: Параметры сцены.
        point_mm: Точка (x, y, z), мм, ниже наблюдателя.

    Returns:
        Координаты (x, y) проекции на экране, мм.
    """
    scale = parameters.observer_z_mm / (parameters.observer_z_mm - point_mm[2])
    return float(point_mm[0] * scale), float(point_mm[1] * scale)
