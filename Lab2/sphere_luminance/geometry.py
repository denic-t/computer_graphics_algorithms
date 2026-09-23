"""Геометрия наблюдения: направления по углам, камера наблюдателя, пересечение лучей со сферой.

Направление (взгляда наблюдателя или оси диаграммы источника) задаётся двумя
углами:

    * наклон psi — угол от вертикали вниз (0° — строго вниз, вдоль -Z);
    * азимут phi — сторона наклона в плоскости XY (0° — к +X, 90° — к +Y).

Единичный вектор направления: (sin psi * cos phi, sin psi * sin phi, -cos psi).

Экран наблюдателя — прямоугольник H x W, перпендикулярный направлению взгляда
и удалённый от наблюдателя на zO. При наклоне 0° он совпадает с плоскостью
z = 0, как в исходной постановке задачи. При повороте взгляда экран
поворачивается вместе с ним (как плоскость кадра камеры): к базису
«вниз, +X, +Y» применяется поворот на угол psi вокруг горизонтальной оси
k = (sin phi, -cos phi, 0) по формуле Родрига.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .models import SceneParameters


def direction_from_angles(tilt_deg: float, azimuth_deg: float) -> np.ndarray:
    """Единичный вектор направления по наклону от вертикали вниз и азимуту.

    Args:
        tilt_deg: Наклон psi от направления -Z, градусы.
        azimuth_deg: Азимут phi в плоскости XY от оси +X к +Y, градусы.

    Returns:
        Вектор (sin psi cos phi, sin psi sin phi, -cos psi).
    """
    tilt = np.radians(tilt_deg)
    azimuth = np.radians(azimuth_deg)
    return np.array(
        [np.sin(tilt) * np.cos(azimuth), np.sin(tilt) * np.sin(azimuth), -np.cos(tilt)]
    )


def tilt_rotation(tilt_deg: float, azimuth_deg: float) -> np.ndarray:
    """Матрица поворота, переводящая направление -Z в direction_from_angles(...).

    Поворот на угол psi вокруг горизонтальной оси k = (sin phi, -cos phi, 0)
    (формула Родрига): R = I cos psi + [k]x sin psi + k k^T (1 - cos psi).

    Args:
        tilt_deg: Наклон psi, градусы.
        azimuth_deg: Азимут phi, градусы.

    Returns:
        Ортогональная матрица 3 x 3.
    """
    tilt = np.radians(tilt_deg)
    azimuth = np.radians(azimuth_deg)
    axis = np.array([np.sin(azimuth), -np.cos(azimuth), 0.0])
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return (
        np.eye(3) * np.cos(tilt)
        + cross * np.sin(tilt)
        + np.outer(axis, axis) * (1.0 - np.cos(tilt))
    )


@dataclass(frozen=True)
class Camera:
    """Наблюдатель с экраном, перпендикулярным направлению взгляда.

    Attributes:
        origin_mm: Положение наблюдателя O, мм.
        forward: Единичное направление взгляда f (на центр экрана).
        screen_x: Единичная ось x экрана (вертикаль изображения, вверх).
        screen_y: Единичная ось y экрана (горизонталь изображения, вправо).
        distance_mm: Расстояние от наблюдателя до экрана, мм (равно zO).
    """

    origin_mm: np.ndarray
    forward: np.ndarray
    screen_x: np.ndarray
    screen_y: np.ndarray
    distance_mm: float

    @staticmethod
    def from_parameters(parameters: SceneParameters) -> Camera:
        """Строит камеру по положению наблюдателя и углам взгляда."""
        rotation = tilt_rotation(parameters.view_tilt_deg, parameters.view_azimuth_deg)
        return Camera(
            origin_mm=np.asarray(parameters.observer_mm, dtype=float),
            forward=rotation @ np.array([0.0, 0.0, -1.0]),
            screen_x=rotation @ np.array([1.0, 0.0, 0.0]),
            screen_y=rotation @ np.array([0.0, 1.0, 0.0]),
            distance_mm=parameters.observer_z_mm,
        )

    def ray_directions(self, screen_x_mm: np.ndarray, screen_y_mm: np.ndarray) -> np.ndarray:
        """Единичные направления лучей из наблюдателя через точки экрана.

        Точка экрана S = O + zO * f + x * e_x + y * e_y, направление d = (S - O)/|S - O|.

        Args:
            screen_x_mm: Координаты x на экране, мм (любая форма).
            screen_y_mm: Координаты y на экране, мм (форма, согласуемая с x).

        Returns:
            Массив направлений формы (..., 3).
        """
        x, y = np.broadcast_arrays(np.asarray(screen_x_mm, float), np.asarray(screen_y_mm, float))
        directions = (
            self.distance_mm * self.forward
            + x[..., np.newaxis] * self.screen_x
            + y[..., np.newaxis] * self.screen_y
        )
        return directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    def project(self, point_mm: np.ndarray) -> tuple[float, float]:
        """Центральная проекция точки пространства на экран.

        Args:
            point_mm: Точка (x, y, z) перед наблюдателем, мм.

        Returns:
            Координаты (x, y) проекции на экране, мм.
        """
        relative = np.asarray(point_mm, float) - self.origin_mm
        scale = self.distance_mm / (relative @ self.forward)
        return float(relative @ self.screen_x * scale), float(relative @ self.screen_y * scale)

    def side_plane_normals(self, height_mm: float, width_mm: float) -> np.ndarray:
        """Внутренние единичные нормали четырёх боковых граней пирамиды видимости.

        Грань проходит через наблюдателя и ребро экрана между соседними углами
        c_i и c_j; её нормаль — c_i x c_j, ориентированная так, чтобы
        направление взгляда f лежало с внутренней стороны.

        Returns:
            Массив 4 x 3.
        """
        half_x, half_y = height_mm / 2.0, width_mm / 2.0
        corners = [
            self.distance_mm * self.forward + sx * half_x * self.screen_x + sy * half_y * self.screen_y
            for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))
        ]
        normals = []
        for index, corner in enumerate(corners):
            normal = np.cross(corner, corners[(index + 1) % 4])
            if normal @ self.forward < 0.0:
                normal = -normal
            normals.append(normal / np.linalg.norm(normal))
        return np.array(normals)


@dataclass(frozen=True)
class SurfaceSample:
    """Видимые точки сферы для набора лучей зрения.

    Attributes:
        hit_mask: True, если луч пересекает сферу.
        points_mm: Координаты точек P на сфере, мм; NaN для лучей без
            пересечения. Последняя ось — (x, y, z).
        normals: Единичные внешние нормали N = (P - C) / R; NaN вне сферы.
    """

    hit_mask: np.ndarray
    points_mm: np.ndarray
    normals: np.ndarray


def pixel_centres_mm(parameters: SceneParameters) -> tuple[np.ndarray, np.ndarray]:
    """Координаты центров пикселей на экране.

    Строка 0 соответствует верхнему краю x = +H/2, столбец 0 — левому краю
    y = -W/2 (как в ЛР1).

    Returns:
        Кортеж (x-координаты строк, y-координаты столбцов), мм.
    """
    x_min_mm, x_max_mm = parameters.x_bounds_mm
    y_min_mm, _ = parameters.y_bounds_mm
    x_coordinates_mm = x_max_mm - (np.arange(parameters.height_px) + 0.5) * parameters.pixel_height_mm
    y_coordinates_mm = y_min_mm + (np.arange(parameters.width_px) + 0.5) * parameters.pixel_width_mm
    return x_coordinates_mm, y_coordinates_mm


def intersect_sphere(parameters: SceneParameters, directions: np.ndarray) -> SurfaceSample:
    """Находит ближайшие к наблюдателю точки пересечения лучей со сферой.

    Луч: X(t) = O + t * d, |d| = 1. Подстановка в |X - C|^2 = R^2 даёт

        t^2 + 2 * b * t + c = 0,  b = d . (O - C),  c = |O - C|^2 - R^2,

    с дискриминантом D = b^2 - c. При D >= 0 ближний корень t = -b - sqrt(D);
    t > 0 означает, что сфера лежит перед наблюдателем.

    Args:
        parameters: Параметры сцены.
        directions: Единичные направления лучей, форма (..., 3).

    Returns:
        Видимые точки сферы и нормали в них.
    """
    observer = np.asarray(parameters.observer_mm, dtype=float)
    center = np.asarray(parameters.sphere_center_mm, dtype=float)
    radius = parameters.sphere_radius_mm

    offset = observer - center
    half_b = directions @ offset
    discriminant = half_b**2 - (offset @ offset - radius**2)

    hit_mask = discriminant >= 0.0
    distances = np.where(hit_mask, -half_b - np.sqrt(np.where(hit_mask, discriminant, 0.0)), np.nan)
    hit_mask &= distances > 0.0

    points_mm = observer + distances[..., np.newaxis] * directions
    points_mm[~hit_mask] = np.nan
    normals = (points_mm - center) / radius

    return SurfaceSample(hit_mask=hit_mask, points_mm=points_mm, normals=normals)
