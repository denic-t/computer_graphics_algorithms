"""Проверки физики, цветовых преобразований и результатов ЛР3."""

from dataclasses import replace
import math

import numpy as np
import pytest

from sphere_color import LightSource, LuminanceService, ParameterValidationError, ParameterValidator, SceneParameters, Spectrum
from sphere_color.colorimetry import RGB_TO_XYZ, XYZ_TO_RGB, load_tables, monitor_images, rgb_to_xyz, spectrum_to_xyz, xyz_to_rgb
from sphere_color.geometry import Camera, angles_to_target, direction_from_angles, tilt_rotation
from sphere_color.physics import LuminanceCalculator
from sphere_color.presets import gaussian_band, neutral_reflectance, white_d65


def test_source_tables_and_color_matrix() -> None:
    tables = load_tables()
    assert tables.wavelengths_nm.tolist() == list(range(380, 781, 10))
    assert tables.xyz_matching[17].tolist() == pytest.approx([0.4334, 0.995, 0.0087])
    assert tables.d65[17] == pytest.approx(104.046)
    np.testing.assert_allclose(RGB_TO_XYZ @ XYZ_TO_RGB, np.eye(3), atol=1e-12)
    sample = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(rgb_to_xyz(xyz_to_rgb(sample)), sample, atol=1e-12)


def test_spectrum_presets_have_expected_shape_and_limits() -> None:
    red = gaussian_band(630, 45).as_array()
    assert int(np.argmax(red)) == (630 - 380) // 10
    assert np.all(red > 0)
    assert np.max(gaussian_band(540, 40, reflectance=True).as_array()) <= 1
    np.testing.assert_allclose(neutral_reflectance().as_array(), 0.7)
    np.testing.assert_allclose(
        white_d65().as_array(), load_tables().d65 / load_tables().d65.max(),
    )


def test_overhead_radiance_matches_lambertian_formula() -> None:
    base = SceneParameters.default()
    flat = Spectrum.from_values([1.0] * 41)
    kd = 0.5
    parameters = replace(
        base,
        reflectance=Spectrum.from_values([kd] * 41),
        lights=(LightSource(0, 0, 2000, 100, flat),),
    )
    ParameterValidator.validate(parameters)
    top_z = parameters.sphere_z_mm + parameters.sphere_radius_mm
    observed = LuminanceCalculator(parameters).luminance(
        np.array([[0.0, 0.0, top_z]]), np.array([[0.0, 0.0, 1.0]])
    )[0]
    distance_m = (2000 - top_z) / 1000
    assert observed == pytest.approx(100 * kd / (math.pi * distance_m**2), rel=1e-12)


def test_white_sources_agree_and_colored_sources_differ() -> None:
    base = SceneParameters.default()
    white = LuminanceService.compute(base.with_white_lights())
    colored = LuminanceService.compute(base)
    np.testing.assert_array_equal(white.image_1, white.image_2)
    assert white.mean_difference == 0
    assert colored.mean_difference > 0
    assert colored.max_difference > 0
    assert colored.field.spectral_w_m2_sr_nm.shape == (400, 400, 41)
    assert colored.field.spectral_w_m2_sr_nm.max() > 0
    assert all(image.dtype == np.uint8 for image in colored.images)


def test_spectral_radiance_scales_with_source_power() -> None:
    base = SceneParameters.default()
    single = replace(base, lights=(base.lights[0],))
    doubled = replace(single, lights=(replace(base.lights[0], intensity_w_sr=2000),))
    first = LuminanceCalculator(single).compute_field().values_w_m2_sr
    second = LuminanceCalculator(doubled).compute_field().values_w_m2_sr
    np.testing.assert_allclose(second, 2 * first, rtol=1e-12)


def test_invalid_spectrum_and_non_square_pixels_are_rejected() -> None:
    base = SceneParameters.default()
    with pytest.raises(ParameterValidationError, match="41"):
        ParameterValidator.validate(replace(base, reflectance=Spectrum.from_values([0.5] * 40)))
    with pytest.raises(ParameterValidationError, match="квадратными"):
        ParameterValidator.validate(replace(base, width_px=500))
    with pytest.raises(ParameterValidationError, match="диапазона"):
        ParameterValidator.validate(replace(base, reflectance=Spectrum.from_values([1.2] * 41)))


def test_export_produces_four_images_and_values(tmp_path) -> None:
    result = LuminanceService.compute(SceneParameters.default())
    paths = LuminanceService.save(result, tmp_path)
    assert {path.name for path in paths} == {
        "image_1.png", "image_2.png", "image_1_gray.png", "image_2_gray.png", "values.txt"
    }
    assert "Центр видимой области" in (tmp_path / "values.txt").read_text(encoding="utf-8")
    from PIL import Image
    with Image.open(tmp_path / "image_1.png") as image:
        assert image.mode == "RGB"
        assert image.size == (400, 400)
    with Image.open(tmp_path / "image_1_gray.png") as image:
        assert image.mode == "L"


def test_xyz_integration_of_flat_spectrum() -> None:
    tables = load_tables()
    expected = np.trapezoid(tables.xyz_matching, dx=10, axis=0)
    np.testing.assert_allclose(spectrum_to_xyz(np.ones(41)), expected)


def test_light_axis_angles_and_fixed_preview_exposure() -> None:
    np.testing.assert_allclose(direction_from_angles(0, 90), [0, 0, -1], atol=1e-15)
    np.testing.assert_allclose(direction_from_angles(90, 0), [1, 0, 0], atol=1e-15)
    np.testing.assert_allclose(direction_from_angles(90, 90), [0, 1, 0], atol=1e-15)

    xyz = np.array([[[0.5, 0.5, 0.5]]])
    normalized = monitor_images(xyz, xyz)[0]
    np.testing.assert_array_equal(normalized, monitor_images(xyz * 0.5, xyz * 0.5)[0])
    fixed = monitor_images(xyz * 0.5, xyz * 0.5, reference_y=0.5)[0]
    assert np.mean(fixed) < np.mean(normalized)


def test_sphere_can_be_partly_outside_the_frame() -> None:
    parameters = replace(SceneParameters.default(), sphere_x_mm=900)
    ParameterValidator.validate(parameters)
    result = LuminanceService.compute(parameters)
    assert result.field.sphere_mask[0].any()
    assert result.statistics.pixel_count > 0
    assert len(result.statistics.probes) == 3
    for point in result.statistics.probes:
        x, y = point.screen_mm
        assert parameters.x_bounds_mm[0] <= x <= parameters.x_bounds_mm[1]
        assert parameters.y_bounds_mm[0] <= y <= parameters.y_bounds_mm[1]
    assert np.isfinite(result.cross_sections[0].values_w_m2_sr).any()
    assert np.isfinite(result.cross_sections[1].values_w_m2_sr).any()


def test_out_of_frame_sphere_reports_no_visible_pixels() -> None:
    parameters = replace(SceneParameters.default(), sphere_x_mm=1000)
    ParameterValidator.validate(parameters)
    with pytest.raises(ValueError, match="ни в один пиксель"):
        LuminanceService.compute(parameters)


def test_full_visibility_is_optional_and_depends_on_view_direction() -> None:
    base = SceneParameters.default()
    ParameterValidator.validate(replace(base, require_full_visibility=True))
    cropped = replace(base, sphere_x_mm=900)
    ParameterValidator.validate(cropped)
    with pytest.raises(ParameterValidationError, match="целиком"):
        ParameterValidator.validate(replace(cropped, require_full_visibility=True))
    tilted = replace(base, view_tilt_deg=10)
    ParameterValidator.validate(tilted)
    with pytest.raises(ParameterValidationError, match="целиком"):
        ParameterValidator.validate(replace(tilted, require_full_visibility=True))


def test_direction_angles_cover_upward_view_and_periodic_azimuth() -> None:
    base = SceneParameters.default()
    angles = angles_to_target(base.lights[0].position_mm, base.sphere_center_mm)
    target = np.asarray(base.sphere_center_mm) - base.lights[0].position_mm
    np.testing.assert_allclose(direction_from_angles(*angles), target / np.linalg.norm(target))
    np.testing.assert_allclose(direction_from_angles(55, 270), direction_from_angles(55, -90))
    np.testing.assert_allclose(tilt_rotation(180, 0), tilt_rotation(180, 123), atol=1e-12)
    np.testing.assert_allclose(angles_to_target((0, 0, 0), (0, 0, 1)), (180, 0))
    ParameterValidator.validate(replace(base, view_azimuth_deg=270))
    ParameterValidator.validate(replace(base, lights=(replace(base.lights[0], axis_azimuth_deg=630),)))
    with pytest.raises(ParameterValidationError, match="конечное"):
        ParameterValidator.validate(replace(base, view_azimuth_deg=float("nan")))

    looking_up = replace(base, view_tilt_deg=180, view_azimuth_deg=270)
    ParameterValidator.validate(looking_up)
    np.testing.assert_allclose(Camera.from_parameters(looking_up).forward, [0, 0, 1], atol=1e-12)

    upward_scene = replace(
        looking_up, height_mm=2000, width_mm=2000, sphere_z_mm=4000,
        require_full_visibility=True,
        lights=(LightSource(0, 0, 3200, 1000, base.lights[0].spectrum, axis_tilt_deg=180),),
    )
    assert LuminanceService.compute(upward_scene).statistics.pixel_count > 0
