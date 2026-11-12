"""Лучевая геометрия и два способа вычисления цвета диффузной сферы."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .colorimetry import rgb_to_xyz, reflectance_to_rgb, spectrum_to_xyz, xyz_to_rgb
from .geometry import Camera, SurfaceSample, intersect_sphere, pixel_centres_mm
from .models import SceneParameters, WAVELENGTHS_NM

MM_PER_M = 1000.0
ROW_BLOCK = 32


@dataclass(frozen=True)
class LuminanceField:
    """Абсолютный спектр и цвет каждого пикселя двух путей расчёта."""

    x_coordinates_mm: np.ndarray
    y_coordinates_mm: np.ndarray
    surface: SurfaceSample
    spectral_w_m2_sr_nm: np.ndarray
    values_w_m2_sr: np.ndarray
    xyz_spectral: np.ndarray
    xyz_rgb: np.ndarray

    @property
    def sphere_mask(self) -> np.ndarray:
        return self.surface.hit_mask


class LuminanceCalculator:
    """Спектральный путь и приближение через RGB-параметры сцены."""

    def __init__(self, parameters: SceneParameters) -> None:
        self.parameters = parameters
        self.reflectance = parameters.reflectance.as_array()
        self.reflectance_rgb = reflectance_to_rgb(self.reflectance)
        self.light_shapes: list[np.ndarray] = []
        self.light_rgb: list[np.ndarray] = []
        for light in parameters.active_lights:
            shape = light.spectrum.as_array()
            integral = np.trapezoid(shape, dx=10.0)
            normalized = shape / integral
            self.light_shapes.append(normalized)
            self.light_rgb.append(xyz_to_rgb(spectrum_to_xyz(light.intensity_w_sr * normalized)))

    def compute_field(self) -> LuminanceField:
        """Трассирует лучи в центре пикселей, блоками по 32 строки."""
        p = self.parameters
        x_mm, y_mm = pixel_centres_mm(p)
        camera = Camera.from_parameters(p)
        shape = (p.height_px, p.width_px)
        spectral = np.zeros((*shape, len(WAVELENGTHS_NM)), dtype=np.float32)
        total = np.zeros(shape)
        xyz_spectral = np.zeros((*shape, 3))
        xyz_rgb = np.zeros((*shape, 3))
        mask = np.zeros(shape, dtype=bool)
        points = np.full((*shape, 3), np.nan)
        normals = np.full((*shape, 3), np.nan)

        for start in range(0, p.height_px, ROW_BLOCK):
            stop = min(start + ROW_BLOCK, p.height_px)
            directions = camera.ray_directions(x_mm[start:stop, None], y_mm[None, :])
            block = intersect_sphere(p, directions)
            mask[start:stop] = block.hit_mask
            points[start:stop] = block.points_mm
            normals[start:stop] = block.normals
            if not block.hit_mask.any():
                continue
            spectrum, rgb = self._at_points(block.points_mm[block.hit_mask], block.normals[block.hit_mask])
            block_spectral = spectral[start:stop]
            block_total = total[start:stop]
            block_xyz_spectral = xyz_spectral[start:stop]
            block_xyz_rgb = xyz_rgb[start:stop]
            block_spectral[block.hit_mask] = spectrum
            block_total[block.hit_mask] = np.trapezoid(spectrum, dx=10.0, axis=-1)
            block_xyz_spectral[block.hit_mask] = spectrum_to_xyz(spectrum)
            block_xyz_rgb[block.hit_mask] = rgb_to_xyz(rgb)

        return LuminanceField(
            x_mm, y_mm, SurfaceSample(mask, points, normals), spectral,
            total, xyz_spectral, xyz_rgb,
        )

    def luminance(self, points_mm: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Абсолютная интегральная радианс в контрольных точках, Вт/(м²·ср)."""
        spectrum, _ = self._at_points(points_mm, normals)
        return np.trapezoid(spectrum, dx=10.0, axis=-1)

    def _at_points(self, points_mm: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.parameters
        positions_m = np.asarray(points_mm) / MM_PER_M
        output_spectrum = np.zeros((len(positions_m), len(WAVELENGTHS_NM)))
        output_rgb = np.zeros((len(positions_m), 3))

        for light, shape, light_rgb in zip(p.active_lights, self.light_shapes, self.light_rgb):
            ray = positions_m - np.asarray(light.position_mm) / MM_PER_M
            squared_distance = np.einsum("ij,ij->i", ray, ray)
            ray_direction = ray / np.sqrt(squared_distance)[:, None]
            cos_emission = np.clip(ray_direction @ light.axis, 0, None)
            cos_incidence = np.clip(-np.einsum("ij,ij->i", ray_direction, normals), 0, None)
            geometry = cos_emission * cos_incidence / (np.pi * squared_distance)
            output_spectrum += (geometry * light.intensity_w_sr)[:, None] * (shape * self.reflectance)
            output_rgb += geometry[:, None] * (light_rgb * self.reflectance_rgb)

        return output_spectrum, output_rgb
