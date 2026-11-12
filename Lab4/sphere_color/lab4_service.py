"""Common geometry/exposure for Lab3 reference paths and reconstructed Lab4 path."""
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
from .analysis import FieldAnalyzer
from .colorimetry import (spectrum_to_xyz, xyz_to_rgb, reflectance_to_rgb,
                         XYZ_TO_LINEAR_SRGB, encode_srgb)
from .models import SceneParameters, ParameterValidator, Spectrum
from .physics import LuminanceCalculator
from .reconstruction import reconstruct


def default_scene():
    return replace(SceneParameters.default().with_white_lights(),
                   reflectance=Spectrum.from_values([0.7]*41), require_full_visibility=True)


def display_images(xyz_fields, reference_y=None):
    reference = reference_y if reference_y is not None else max(float(v[..., 1].max()) for v in xyz_fields)
    if not np.isfinite(reference) or reference < 0:
        raise ValueError("Некорректная экспозиция.")
    reference = max(reference, 1e-15)  # An unlit visible sphere produces a valid black image.
    return tuple(np.rint(255 * encode_srgb(np.clip(v @ XYZ_TO_LINEAR_SRGB.T / reference, 0, 1))).astype(np.uint8)
                 for v in xyz_fields)


def split_image(left, right, fraction=0.5):
    if left.shape != right.shape or not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Неверная пара изображений или положение разделителя.")
    x = round(left.shape[1] * fraction)
    result = right.copy()
    result[:, :x] = left[:, :x]
    return result


@dataclass
class Lab4Result:
    parameters: SceneParameters
    field: object
    original_field: object
    spectra: dict
    image_A: np.ndarray
    image_1: np.ndarray
    image_2: np.ndarray
    statistics: object
    cross_sections: tuple
    metrics: dict
    report: str
    method: str

    @property
    def xyz_fields(self):
        return self.field.xyz_spectral, self.original_field.xyz_spectral, self.original_field.xyz_rgb

    @property
    def images(self):
        grey = np.rint(255*encode_srgb(np.clip(self.field.xyz_spectral[..., 1] /
                         max(max(float(v[..., 1].max()) for v in self.xyz_fields), 1e-15), 0, 1))).astype(np.uint8)
        return self.image_A, self.image_1, self.image_2, grey

    @property
    def mean_difference(self):
        return self.metrics["A_vs_1"]["display_mae"]


class Lab4Service:
    @staticmethod
    def compute(p, method="bounded", rgb_overrides=None):
        ParameterValidator.validate(p)
        overrides = rgb_overrides or {}
        original = LuminanceCalculator(p).compute_field()
        calc = LuminanceCalculator(p)
        spectra = {}
        def recover(key, values, reflectance=False):
            rgb = reflectance_to_rgb(values) if reflectance else xyz_to_rgb(spectrum_to_xyz(values))
            rec = reconstruct(overrides.get(key, rgb), reflectance=reflectance, method=method)
            spectra[key] = {"original": values.copy(), "reconstruction": rec,
                            "unit": "коэффициент отражения" if reflectance else "1/нм; исходный спектр нормирован по интегралу",
                            "rmse": float(np.sqrt(np.mean((values-rec.values)**2))),
                            "max_error": float(np.max(np.abs(values-rec.values)))}
            return rec.values
        calc.reflectance = recover("surface", p.reflectance.as_array(), True)
        # Keep original integrated I0; never normalize reconstructed spectra again:
        # doing so would silently change their XYZ and absolute brightness.
        active_index = 0
        for index, light in enumerate(p.lights, 1):
            if not light.enabled:
                continue
            values = light.spectrum.as_array()
            normalized = values / np.trapezoid(values, dx=10.0)
            calc.light_shapes[active_index] = recover(f"light_{index}", normalized)
            spectra[f"light_{index}"]["intensity_w_sr"] = light.intensity_w_sr
            active_index += 1
        restored = calc.compute_field()
        analyzer = FieldAnalyzer()
        statistics = analyzer.collect_statistics(restored)
        xyz_fields = (restored.xyz_spectral, original.xyz_spectral, original.xyz_rgb)
        images = display_images(xyz_fields)
        metrics = {}
        mask = original.sphere_mask
        for i, name in ((1, "A_vs_1"), (2, "A_vs_2")):
            delta = images[0][mask].astype(float) - images[i][mask].astype(float)
            xyz_delta = xyz_fields[0][mask] - xyz_fields[i][mask]
            metrics[name] = {"display_mae": float(np.abs(delta).mean()),
                             "display_rmse": float(np.sqrt(np.mean(delta**2))),
                             "display_max": float(np.abs(delta).max()),
                             "absolute_xyz_rmse": float(np.sqrt(np.mean(xyz_delta**2)))}
        lines = ["ЛР4: image_A — восстановление; image_1 — спектральная ЛР3; image_2 — RGB ЛР3.",
                 f"Метод: {method}. RGB: линейный CIE 1931, монитор: sRGB D65.",
                 "Общая нормировка трёх изображений по максимальному Y.",
                 "Абсолютная интегральная энергетическая яркость image_A, Вт/(м²·ср):"]
        for point in (*statistics.probes, statistics.minimum, statistics.maximum):
            lines.append(f"{point.title}: P={tuple(round(v, 3) for v in point.point_mm)} мм; L={point.luminance_w_m2_sr:.9e}")
        # Photometric luminance has a distinct definition and units.
        y = restored.xyz_spectral[..., 1][mask] * 683.0
        lines.append(f"Фотометрическая яркость: min={y.min():.9e}, max={y.max():.9e} кд/м² (683·Y).")
        for key, data in spectra.items():
            rec = data["reconstruction"]
            lines.append(f"{key}: RMSE спектра={data['rmse']:.6g}; относительная ошибка XYZ={rec.xyz_relative_error:.6g}; отрицательных отсчётов={rec.negative_samples}")
            if "intensity_w_sr" in data:
                lines.append(f"  Интегральная сила излучения: исходная={data['intensity_w_sr']:.8g}, восстановленная={data['intensity_w_sr']*np.trapezoid(rec.values,dx=10):.8g} Вт/ср.")
        if any(np.any(v["reconstruction"].values > 1+1e-9) for k,v in spectra.items() if k == "surface"):
            lines.append("ВНИМАНИЕ: отражение >1 в точном базисном методе; результат физически недопустим.")
        if any(v["reconstruction"].negative_samples for v in spectra.values()):
            lines.append("ВНИМАНИЕ: точный метод дал отрицательные спектры; это математическое, а не физическое решение.")
        for key, values in metrics.items():
            lines.append(key + ": " + "; ".join(f"{k}={v:.7g}" for k,v in values.items()))
        return Lab4Result(p, restored, original, spectra, *images, statistics,
                          analyzer.extract_cross_sections(restored), metrics, "\n".join(lines), method)

    @staticmethod
    def save(result, directory, fraction=0.5):
        from .exporting import export_result
        return export_result(result, Path(directory), fraction)
