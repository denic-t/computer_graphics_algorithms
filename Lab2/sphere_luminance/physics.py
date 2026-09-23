"""Расчёт яркости поверхности сферы по модели Блинн-Фонга.

Рабочие формулы (обозначения соответствуют рисунку задания):

    s = P - P_L,                         R^2 = |s|^2,
    cos(theta) = s . O_L / |s|,          O_L — единичная ось диаграммы источника,
    I(s) = I0 * cos(theta),              ламбертовская диаграмма излучения,
    cos(sigma) = -s . N / |s|,           угол падения на поверхность,
    E(P) = I(s) * cos(sigma) / R^2,      освещённость от одного источника,
    v = (O - P) / |O - P|,               направление на наблюдателя,
    h = (v + l) / |v + l|,  l = -s/|s|,  вектор полупути,
    f = kd + ks * (h . N)^ke,            функция отражения Блинн-Фонга,
    L(P, v) = 1/pi * sum_i E_i(P) * f_i  яркость, Вт/(м^2 * ср).

Лучи зрения строятся камерой наблюдателя (geometry.Camera) с учётом
поворота взгляда. Отрицательные cos(theta) и cos(sigma) заменяются нулём:
источник не излучает назад относительно своей оси, а точки сферы, отвёрнутые от источника,
находятся в собственной тени. Расстояния переводятся из мм в метры, чтобы
освещённость получалась в Вт/м^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Camera, SurfaceSample, intersect_sphere, pixel_centres_mm
from .models import LightSource, SceneParameters

MILLIMETRES_PER_METRE = 1000.0


@dataclass(frozen=True)
class LuminanceField:
    """Распределение яркости по растру изображения.

    Attributes:
        x_coordinates_mm: x-координаты центров строк экрана, мм.
        y_coordinates_mm: y-координаты центров столбцов экрана, мм.
        surface: Видимые точки сферы для каждого пикселя.
        values_w_m2_sr: Яркость L, Вт/(м^2*ср); вне сферы — 0.
    """

    x_coordinates_mm: np.ndarray
    y_coordinates_mm: np.ndarray
    surface: SurfaceSample
    values_w_m2_sr: np.ndarray

    @property
    def sphere_mask(self) -> np.ndarray:
        """Маска пикселей, в которых видна сфера."""
        return self.surface.hit_mask


class LuminanceCalculator:
    """Вычислитель яркости сферы для одного набора параметров сцены."""

    def __init__(self, parameters: SceneParameters) -> None:
        """Сохраняет параметры сцены для последующих расчётов.

        Args:
            parameters: Проверенный набор исходных данных.
        """
        self._parameters = parameters

    def compute_field(self) -> LuminanceField:
        """Рассчитывает яркость во всех пикселях изображения.

        Лучи строятся через центры пикселей; яркость вычисляется только в
        точках, где луч попал на сферу, остальные пиксели остаются нулевыми.

        Returns:
            Поле яркости по растру Hres x Wres.
        """
        x_coordinates_mm, y_coordinates_mm = pixel_centres_mm(self._parameters)
        directions = Camera.from_parameters(self._parameters).ray_directions(
            x_coordinates_mm[:, np.newaxis], y_coordinates_mm[np.newaxis, :]
        )
        surface = intersect_sphere(self._parameters, directions)

        values = np.zeros(surface.hit_mask.shape)
        values[surface.hit_mask] = self.luminance(
            surface.points_mm[surface.hit_mask], surface.normals[surface.hit_mask]
        )

        return LuminanceField(
            x_coordinates_mm=x_coordinates_mm,
            y_coordinates_mm=y_coordinates_mm,
            surface=surface,
            values_w_m2_sr=values,
        )

    def luminance(self, points_mm: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Вычисляет яркость в точках сферы от всех включённых источников.

        Args:
            points_mm: Точки на сфере, форма (..., 3), мм.
            normals: Единичные нормали в этих точках, форма (..., 3).

        Returns:
            Яркость L, Вт/(м^2*ср), форма (...).
        """
        points_m = points_mm / MILLIMETRES_PER_METRE
        observer_m = np.asarray(self._parameters.observer_mm) / MILLIMETRES_PER_METRE
        to_observer = _normalize(observer_m - points_m)

        total = np.zeros(points_m.shape[:-1])
        for light in self._parameters.active_lights:
            total += self._contribution(light, points_m, normals, to_observer)

        return total / np.pi

    def _contribution(
        self,
        light: LightSource,
        points_m: np.ndarray,
        normals: np.ndarray,
        to_observer: np.ndarray,
    ) -> np.ndarray:
        """Слагаемое E_i(P) * f_i одного источника (без множителя 1/pi).

        Args:
            light: Источник света.
            points_m: Точки на сфере, м.
            normals: Единичные нормали.
            to_observer: Единичные векторы v на наблюдателя.

        Returns:
            Произведение освещённости на функцию отражения.
        """
        light_m = np.asarray(light.position_mm) / MILLIMETRES_PER_METRE
        ray = points_m - light_m
        distance_squared = np.einsum("...i,...i->...", ray, ray)
        ray_direction = ray / np.sqrt(distance_squared)[..., np.newaxis]

        cos_theta = np.clip(ray_direction @ light.axis, 0.0, None)
        cos_sigma = np.clip(-np.einsum("...i,...i->...", ray_direction, normals), 0.0, None)
        illuminance = light.intensity_w_sr * cos_theta * cos_sigma / distance_squared

        half_vector = _normalize(to_observer - ray_direction)
        cos_half = np.clip(np.einsum("...i,...i->...", half_vector, normals), 0.0, None)

        material = self._parameters.material
        reflectance = material.diffuse + material.specular * cos_half**material.shininess

        return illuminance * reflectance


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Нормирует векторы по последней оси."""
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
