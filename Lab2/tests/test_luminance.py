"""Проверки расчёта яркости на сфере и ограничений параметров."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

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
    assert result.statistics.maximum.luminance_w_m2_sr == pytest.approx(expected, rel=1e-3)


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
        {"sphere_x_mm": 400.0},
        {"sphere_radius_mm": 450.0},
        {"observer_z_mm": 600.0},
        {"width_px": 500},
        {"material": BlinnPhongMaterial(diffuse=0.3, specular=0.7, shininess=0.0)},
        {"lights": ()},
    ],
)
def test_invalid_parameters_are_rejected(changes: dict) -> None:
    with pytest.raises(ParameterValidationError):
        ParameterValidator.validate(replace(SceneParameters.default(), **changes))
