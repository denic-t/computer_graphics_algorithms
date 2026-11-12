"""Reproducible scene, absolute arrays, tables and report-ready figures."""
import csv
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from .colorimetry import xyz_to_rgb
from .models import SceneParameters, LightSource, Spectrum, ParameterValidator, WAVELENGTHS_NM


def read_scene(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    data = payload["scene"]
    data["reflectance"] = Spectrum.from_values(data["reflectance"]["values"])
    data["lights"] = tuple(LightSource(**{**v, "spectrum": Spectrum.from_values(v["spectrum"]["values"])}) for v in data["lights"])
    scene = SceneParameters(**data)
    ParameterValidator.validate(scene)
    return scene, payload.get("method", "bounded"), payload.get("rgb_inputs", {})


def spectrum_display_delta(data):
    original = data["original"]
    difference = data["reconstruction"].values - original
    tolerance = max(float(np.max(np.abs(original))), 1e-15) * 1e-12
    return np.where(np.abs(difference) <= tolerance, 0.0, difference)


def spectrum_figure(key, data):
    figure = Figure(figsize=(8, 5), dpi=120, layout="constrained")
    ax, delta = figure.subplots(2, 1, sharex=True)
    original, recovered = data["original"], data["reconstruction"].values
    ax.plot(WAVELENGTHS_NM, original, label="Исходный")
    ax.plot(WAVELENGTHS_NM, recovered, "--", label="Восстановленный")
    title = "Поверхность" if key == "surface" else f"Источник {key.split('_')[-1]}"
    ax.set(title=title, ylabel="Отражение" if key == "surface" else "Спектр, 1/нм")
    lower = min(0, float(min(original.min(), recovered.min()))*1.05)
    upper = max(float(max(original.max(), recovered.max()))*1.05, 1e-12)
    ax.set_ylim(lower, upper)
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.legend(); ax.grid(alpha=.3)
    difference = spectrum_display_delta(data)
    delta.plot(WAVELENGTHS_NM, difference, color="#b44c40")
    if not np.any(difference):
        span = max(float(np.max(np.abs(original))) * .01, 1e-12)
        delta.set_ylim(-span, span)
    delta.axhline(0, color="gray", linewidth=.6)
    delta.set(xlabel="Длина волны, нм", ylabel="Отклонение")
    delta.grid(alpha=.3)
    return figure


def export_result(result, directory, fraction=.5):
    from .lab4_service import split_image
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    def record(name):
        path = directory/name; paths.append(path); return path
    for name, pixels in zip(("image_A", "image_1", "image_2", "image_A_gray"), result.images):
        Image.fromarray(pixels).save(record(name+".png"))
    for name, right in (("1", result.image_1), ("2", result.image_2)):
        # Keep pixels unannotated; filename and metrics identify each side.
        Image.fromarray(split_image(result.image_A, right, fraction)).save(record(f"split_A_vs_{name}.png"))
        Image.fromarray(np.abs(result.image_A.astype(np.int16)-right.astype(np.int16)).astype(np.uint8)).save(record(f"difference_A_vs_{name}.png"))
    for key, data in result.spectra.items():
        rec = data["reconstruction"]
        with record(f"spectrum_{key}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("wavelength_nm", "original", "reconstructed", "delta", "unit"))
            writer.writerows((nm, a, b, b-a, data["unit"]) for nm,a,b in zip(WAVELENGTHS_NM, data["original"], rec.values))
        fig = spectrum_figure(key, data)
        fig.savefig(record(f"spectrum_{key}.png"))
        fig.savefig(record(f"spectrum_{key}.svg"))
        fig.clear()
    with record("brightness.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("point", "x_mm", "y_mm", "z_mm", "radiance_W_m2_sr", "luminance_cd_m2"))
        field = result.field
        for point in (*result.statistics.probes, result.statistics.minimum, result.statistics.maximum):
            ij = np.unravel_index(np.nanargmin(np.sum((field.surface.points_mm-np.asarray(point.point_mm))**2, axis=-1)), field.sphere_mask.shape)
            writer.writerow((point.title, *point.point_mm, point.luminance_w_m2_sr, 683*field.xyz_spectral[ij][1]))
    record("values.txt").write_text(result.report+"\n", encoding="utf-8")
    record("metrics.json").write_text(json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    record("scene.json").write_text(json.dumps({"scene": asdict(result.parameters), "method": result.method,
        "rgb_inputs": {k:v["reconstruction"].target_rgb.tolist() for k,v in result.spectra.items()},
        "split_fraction": fraction}, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(record("absolute_fields.npz"), wavelengths_nm=WAVELENGTHS_NM,
        sphere_mask=result.field.sphere_mask, points_mm=result.field.surface.points_mm,
        spectral_A_W_m2_sr_nm=result.field.spectral_w_m2_sr_nm,
        spectral_1_W_m2_sr_nm=result.original_field.spectral_w_m2_sr_nm,
        radiance_A_W_m2_sr=result.field.values_w_m2_sr,
        xyz_A=result.field.xyz_spectral, xyz_1=result.original_field.xyz_spectral,
        xyz_2=result.original_field.xyz_rgb, rgb_A=xyz_to_rgb(result.field.xyz_spectral),
        rgb_1=xyz_to_rgb(result.original_field.xyz_spectral), rgb_2=xyz_to_rgb(result.original_field.xyz_rgb))
    return tuple(paths)
