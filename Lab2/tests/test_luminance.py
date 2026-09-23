"""Проверки расчёта яркости на сфере и ограничений параметров."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from sphere_luminance.geometry import Camera, direction_from_angles, tilt_rotation
from sphere_luminance.models import square_pixel_update
from sphere_luminance.physics import LuminanceCalculator
from sphere_luminance import (
    BlinnPhongMaterial,
    LightSource,
    LuminanceService,
    ParameterValidationError,
    ParameterValidator,
    SceneParameters,
)


def overhead_scene(intensity_w_sr: float = 100.0) -> SceneParameters:
    """Сцена с одним источником строго над центром сферы."""
    return replace(
        SceneParameters.default(),
        lights=(LightSource(x_mm=0.0, y_mm=0.0, z_mm=2000.0, intensity_w_sr=intensity_w_sr),),
    )


def test_default_parameters_are_valid() -> None:
    ParameterValidator.validate(SceneParameters.default())


def test_top_point_matches_analytic_value() -> None:
    """В вершине сферы cos θ = cos σ = h·N = 1, поэтому L = I0 / d² · (kd + ks) / π."""
    parameters = overhead_scene()
    result = LuminanceService.compute(parameters)

    top_z_m = (parameters.sphere_z_mm + parameters.sphere_radius_mm) / 1000.0
    distance_m = 2.0 - top_z_m
    material = parameters.material
    expected = 100.0 / distance_m**2 * (material.diffuse + material.specular) / math.pi

    top_probe = result.statistics.probes[0]
    assert top_probe.point_mm == pytest.approx((0.0, 0.0, top_z_m * 1000.0))
    assert top_probe.luminance_w_m2_sr == pytest.approx(expected, rel=1e-12)
    # Максимум по растру берётся в центре ближайшего к вершине пикселя, отсюда допуск.
    assert result.statistics.maximum.luminance_w_m2_sr == pytest.approx(expected, rel=5e-3)


def test_luminance_is_linear_in_intensity() -> None:
    single = LuminanceService.compute(overhead_scene(100.0)).field.values_w_m2_sr
    double = LuminanceService.compute(overhead_scene(200.0)).field.values_w_m2_sr
    np.testing.assert_allclose(double, 2.0 * single)


def test_normalization_and_background() -> None:
    result = LuminanceService.compute(SceneParameters.default())
    assert result.grey_levels.dtype == np.uint8
    assert result.grey_levels.max() == 255
    assert not result.grey_levels[~result.field.sphere_mask].any()


def test_side_light_leaves_far_side_dark() -> None:
    """Точки, отвёрнутые от источника, находятся в собственной тени."""
    parameters = replace(
        SceneParameters.default(),
        lights=(LightSource(x_mm=0.0, y_mm=8000.0, z_mm=500.0, intensity_w_sr=1000.0),),
    )
    statistics = LuminanceService.compute(parameters).statistics
    assert statistics.minimum.luminance_w_m2_sr == 0.0
    assert statistics.minimum.point_mm[1] < parameters.sphere_y_mm


def test_disabled_light_is_ignored() -> None:
    base = SceneParameters.default()
    first_only = replace(base, lights=base.lights[:1])
    with_disabled = replace(base, lights=(base.lights[0], replace(base.lights[1], enabled=False)))
    np.testing.assert_array_equal(
        LuminanceService.compute(first_only).field.values_w_m2_sr,
        LuminanceService.compute(with_disabled).field.values_w_m2_sr,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"sphere_x_mm": 500.0},
        {"sphere_radius_mm": 450.0},
        {"view_tilt_deg": 20.0},
        {"view_tilt_deg": 95.0},
        {"view_azimuth_deg": 200.0},
        {"observer_z_mm": 600.0},
        {"width_px": 500},
        {"material": BlinnPhongMaterial(diffuse=0.3, specular=0.7, shininess=0.0)},
        {"lights": ()},
    ],
)
def test_invalid_parameters_are_rejected(changes: dict) -> None:
    with pytest.raises(ParameterValidationError):
        ParameterValidator.validate(replace(SceneParameters.default(), **changes))


def light_with_axis(tilt_deg: float, azimuth_deg: float = 0.0) -> LightSource:
    return LightSource(
        x_mm=0.0, y_mm=0.0, z_mm=2000.0, intensity_w_sr=100.0,
        axis_tilt_deg=tilt_deg, axis_azimuth_deg=azimuth_deg,
    )


@pytest.mark.parametrize("tilt, azimuth", [(0.0, 0.0), (30.0, 45.0), (60.0, -120.0), (89.0, 180.0)])
def test_rotation_maps_down_to_direction(tilt: float, azimuth: float) -> None:
    rotation = tilt_rotation(tilt, azimuth)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rotation @ [0.0, 0.0, -1.0], direction_from_angles(tilt, azimuth), atol=1e-12)


def test_direction_angles_convention() -> None:
    np.testing.assert_allclose(direction_from_angles(0.0, 123.0), [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(direction_from_angles(90.0, 0.0), [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(direction_from_angles(90.0, 90.0), [0.0, 1.0, 0.0], atol=1e-12)


def test_untilted_camera_looks_through_screen_plane() -> None:
    """При наклоне 0° экран лежит в плоскости z = 0: луч через (x, y) идёт в точку (x, y, 0)."""
    parameters = SceneParameters.default()
    camera = Camera.from_parameters(parameters)
    direction = camera.ray_directions(np.array(120.0), np.array(-300.0))
    expected = np.array([120.0, -300.0, -parameters.observer_z_mm])
    np.testing.assert_allclose(direction, expected / np.linalg.norm(expected))
    assert camera.project(np.array([120.0, -300.0, 0.0])) == pytest.approx((120.0, -300.0))


def test_view_rotation_moves_sphere_but_keeps_luminance() -> None:
    """Поворот взгляда меняет только положение сферы на кадре, а не её яркость."""
    straight = LuminanceService.compute(SceneParameters.default()).statistics
    turned = LuminanceService.compute(
        replace(SceneParameters.default(), view_tilt_deg=8.0, view_azimuth_deg=0.0)
    ).statistics

    center_x, center_y = turned.probes[0].screen_mm
    assert center_x < -300.0 and center_y == pytest.approx(0.0, abs=1e-6)
    assert turned.probes[0].luminance_w_m2_sr == pytest.approx(straight.probes[0].luminance_w_m2_sr)
    assert turned.maximum.luminance_w_m2_sr == pytest.approx(straight.maximum.luminance_w_m2_sr, rel=2e-2)


def test_light_axis_follows_lambert_cosine() -> None:
    """В вершине сферы под источником L пропорциональна cos угла между осью и лучом."""
    straight = replace(SceneParameters.default(), lights=(light_with_axis(0.0),))
    tilted = replace(SceneParameters.default(), lights=(light_with_axis(60.0, 30.0),))
    top_straight = LuminanceService.compute(straight).statistics.probes[0].luminance_w_m2_sr
    top_tilted = LuminanceService.compute(tilted).statistics.probes[0].luminance_w_m2_sr
    assert top_tilted == pytest.approx(0.5 * top_straight, rel=1e-12)


def test_light_aimed_at_sphere_is_brighter_than_aimed_away() -> None:
    base = SceneParameters.default()
    source = LightSource(x_mm=0.0, y_mm=1500.0, z_mm=1500.0, intensity_w_sr=100.0)
    toward = replace(source, axis_tilt_deg=54.0, axis_azimuth_deg=-90.0)
    away = replace(source, axis_tilt_deg=54.0, axis_azimuth_deg=90.0)
    field_toward = LuminanceCalculator(replace(base, lights=(toward,))).compute_field()
    field_away = LuminanceCalculator(replace(base, lights=(away,))).compute_field()
    assert field_toward.values_w_m2_sr.sum() > 5.0 * field_away.values_w_m2_sr.sum()


def test_light_pointing_up_does_not_illuminate() -> None:
    parameters = replace(SceneParameters.default(), lights=(light_with_axis(180.0),))
    with pytest.raises(ValueError, match="не освещена"):
        LuminanceService.compute(parameters)


BASE_PIXELS = {"height_mm": 1500.0, "width_mm": 1000.0, "height_px": 600, "width_px": 400}


@pytest.mark.parametrize(
    "edited, value, expected",
    [
        ("height_mm", 3000.0, {"width_mm": 2000.0}),
        ("width_mm", 500.0, {"height_mm": 750.0}),
        ("height_px", 300, {"width_px": 200}),
        ("width_px", 800, {"height_px": 1200}),
        ("height_px", 301, {"width_px": 201, "width_mm": pytest.approx(201 * 1500.0 / 301)}),
        ("height_px", 0, {}),
    ],
)
def test_square_pixel_update(edited: str, value: float, expected: dict) -> None:
    values = {**BASE_PIXELS, edited: value}
    updates = square_pixel_update(edited, values)
    assert updates == expected
    if updates:
        merged = {**values, **updates}
        assert merged["height_mm"] / merged["height_px"] == pytest.approx(
            merged["width_mm"] / merged["width_px"]
        )
