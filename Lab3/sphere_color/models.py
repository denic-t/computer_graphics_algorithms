"""Параметры спектральной сцены и проверка геометрических ограничений ЛР3."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections.abc import Iterable

import numpy as np

from .geometry import Camera, direction_from_angles

WAVELENGTHS_NM = tuple(range(380, 781, 10))
MAXIMUM_LIGHT_COUNT = 6


class ParameterValidationError(ValueError):
    """Некорректные параметры сцены или спектральной таблицы."""


@dataclass(frozen=True)
class Spectrum:
    """Таблица из 41 значения на фиксированной сетке 380–780 нм."""

    values: tuple[float, ...]

    @classmethod
    def from_values(cls, values: Iterable[float]) -> Spectrum:
        return cls(tuple(float(value) for value in values))

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=float)


@dataclass(frozen=True)
class LightSource:
    """Точечный источник с ламбертовской диаграммой и спектром мощности."""

    x_mm: float
    y_mm: float
    z_mm: float
    intensity_w_sr: float
    spectrum: Spectrum
    axis_tilt_deg: float = 0.0
    axis_azimuth_deg: float = 0.0
    enabled: bool = True

    @property
    def position_mm(self) -> tuple[float, float, float]:
        return self.x_mm, self.y_mm, self.z_mm

    @property
    def axis(self) -> np.ndarray:
        return direction_from_angles(self.axis_tilt_deg, self.axis_azimuth_deg)


@dataclass(frozen=True)
class SceneParameters:
    """Экран, наблюдатель, сфера, её спектральное отражение и источники."""

    height_mm: float
    width_mm: float
    height_px: int
    width_px: int
    observer_z_mm: float
    view_tilt_deg: float
    view_azimuth_deg: float
    sphere_x_mm: float
    sphere_y_mm: float
    sphere_z_mm: float
    sphere_radius_mm: float
    reflectance: Spectrum
    lights: tuple[LightSource, ...] = field(default_factory=tuple)
    require_full_visibility: bool = False

    @property
    def pixel_height_mm(self) -> float:
        return self.height_mm / self.height_px

    @property
    def pixel_width_mm(self) -> float:
        return self.width_mm / self.width_px

    @property
    def x_bounds_mm(self) -> tuple[float, float]:
        return -self.height_mm / 2, self.height_mm / 2

    @property
    def y_bounds_mm(self) -> tuple[float, float]:
        return -self.width_mm / 2, self.width_mm / 2

    @property
    def observer_mm(self) -> tuple[float, float, float]:
        return 0.0, 0.0, self.observer_z_mm

    @property
    def sphere_center_mm(self) -> tuple[float, float, float]:
        return self.sphere_x_mm, self.sphere_y_mm, self.sphere_z_mm

    @property
    def active_lights(self) -> tuple[LightSource, ...]:
        return tuple(light for light in self.lights if light.enabled)

    @staticmethod
    def default() -> SceneParameters:
        """Цветная демонстрационная сцена с двумя разноцветными источниками."""
        from .colorimetry import load_tables

        tables = load_tables()
        wl = np.asarray(WAVELENGTHS_NM)
        d65 = tables.d65
        red = d65 * (0.02 + np.exp(-0.5 * ((wl - 630) / 45) ** 2))
        blue = d65 * (0.02 + np.exp(-0.5 * ((wl - 465) / 38) ** 2))
        reflectance = 0.12 + 0.65 * np.exp(-0.5 * ((wl - 540) / 105) ** 2)
        return SceneParameters(
            height_mm=1500, width_mm=1500, height_px=400, width_px=400,
            observer_z_mm=3000, view_tilt_deg=0, view_azimuth_deg=0,
            sphere_x_mm=0, sphere_y_mm=0, sphere_z_mm=400, sphere_radius_mm=250,
            reflectance=Spectrum.from_values(reflectance),
            require_full_visibility=False,
            lights=(
                LightSource(-900, 700, 2000, 1000, Spectrum.from_values(red), 35, -38),
                LightSource(800, -900, 1200, 500, Spectrum.from_values(blue), 56, 132),
            ),
        )

    def with_white_lights(self) -> SceneParameters:
        """Сохраняет геометрию и мощности, заменяя формы спектров на D65."""
        from .colorimetry import load_tables

        white = Spectrum.from_values(load_tables().d65)
        return replace(self, lights=tuple(replace(light, spectrum=white) for light in self.lights))


class ParameterValidator:
    """Проверяет параметры и, по выбору пользователя, полное попадание в кадр."""

    @staticmethod
    def validate(p: SceneParameters) -> None:
        def check(value: float, low: float, high: float, title: str) -> None:
            if not np.isfinite(value) or not low <= value <= high:
                raise ParameterValidationError(f"{title}: требуется значение от {low:g} до {high:g}.")

        def finite(value: float, title: str) -> None:
            if not np.isfinite(value):
                raise ParameterValidationError(f"{title}: требуется конечное число градусов.")

        for value, title in ((p.height_mm, "H"), (p.width_mm, "W")):
            check(value, 100, 10000, title)
        for value, title in ((p.height_px, "Hres"), (p.width_px, "Wres")):
            if int(value) != value:
                raise ParameterValidationError(f"{title}: требуется целое число пикселей.")
            check(value, 200, 800, title)
        check(p.observer_z_mm, 100, 100000, "zO")
        check(p.view_tilt_deg, 0, 180, "Наклон взгляда")
        finite(p.view_azimuth_deg, "Азимут взгляда")
        for value, title in ((p.sphere_x_mm, "xC"), (p.sphere_y_mm, "yC")):
            check(value, -10000, 10000, title)
        check(p.sphere_z_mm, 0, 10000, "zC")
        check(p.sphere_radius_mm, 0.001, 10000, "Радиус сферы")
        if abs(p.pixel_height_mm - p.pixel_width_mm) / max(p.pixel_height_mm, p.pixel_width_mm) > 1e-3:
            raise ParameterValidationError("Пиксели должны быть квадратными: H/Hres = W/Wres.")

        ParameterValidator._spectrum(p.reflectance, "Отражение", maximum=1)
        if not 1 <= len(p.lights) <= MAXIMUM_LIGHT_COUNT or not p.active_lights:
            raise ParameterValidationError(f"Нужен хотя бы один включённый источник, всего не более {MAXIMUM_LIGHT_COUNT}.")
        for i, light in enumerate(p.lights, 1):
            for value, title in ((light.x_mm, "xL"), (light.y_mm, "yL")):
                check(value, -10000, 10000, f"Источник {i}: {title}")
            check(light.z_mm, -10000, 10000, f"Источник {i}: zL")
            check(light.intensity_w_sr, 0.01, 10000, f"Источник {i}: I0")
            check(light.axis_tilt_deg, 0, 180, f"Источник {i}: наклон")
            finite(light.axis_azimuth_deg, f"Источник {i}: азимут")
            ParameterValidator._spectrum(light.spectrum, f"Источник {i}: спектр")
            if np.linalg.norm(np.asarray(light.position_mm) - p.sphere_center_mm) <= p.sphere_radius_mm:
                raise ParameterValidationError(f"Источник {i} должен находиться вне сферы.")

        offset = np.asarray(p.sphere_center_mm) - np.asarray(p.observer_mm)
        if np.linalg.norm(offset) <= p.sphere_radius_mm:
            raise ParameterValidationError("Наблюдатель должен находиться вне сферы.")
        if p.require_full_visibility:
            camera = Camera.from_parameters(p)
            if offset @ camera.forward <= p.sphere_radius_mm:
                raise ParameterValidationError(
                    "Сфера не помещается целиком перед наблюдателем. "
                    "Отключите требование полного попадания или измените направление взгляда."
                )
            distances = camera.side_plane_normals(p.height_mm, p.width_mm) @ offset
            if np.min(distances) < p.sphere_radius_mm:
                raise ParameterValidationError(
                    "Сфера не помещается целиком в кадре. "
                    "Отключите требование полного попадания или измените сцену."
                )

    @staticmethod
    def _spectrum(spectrum: Spectrum, title: str, maximum: float | None = None) -> None:
        values = spectrum.as_array()
        if values.shape != (len(WAVELENGTHS_NM),) or not np.all(np.isfinite(values)):
            raise ParameterValidationError(f"{title}: требуются 41 конечное значение для 380–780 нм.")
        if np.any(values < 0) or (maximum is not None and np.any(values > maximum)):
            raise ParameterValidationError(f"{title}: значения вне допустимого диапазона.")
        if not np.any(values > 0):
            raise ParameterValidationError(f"{title}: спектр не может быть нулевым.")
