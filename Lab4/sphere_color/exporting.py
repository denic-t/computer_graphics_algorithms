"""Reproducible scene, absolute arrays, tables and report-ready figures."""
import csv
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from .colorimetry import xyz_to_rgb, rgb_to_xyz, spectrum_to_xyz, load_tables
from .reconstruction import reconstruct, integration_matrix, METHODS
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


def spectrum_details(key, data):
    reflectance = key == "surface"
    rec = data["reconstruction"]
    values = rec.values
    tables = load_tables()
    a = integration_matrix()
    if reflectance:
        target = rgb_to_xyz(rec.target_rgb * xyz_to_rgb(spectrum_to_xyz(tables.d65)))
        a = a * tables.d65
    else:
        target = rgb_to_xyz(rec.target_rgb)
    above = values > 1 + 1e-12 if reflectance else np.zeros(values.shape, bool)
    return {"target_xyz": target, "actual_xyz": a @ values,
            "xyz_relative_error": rec.xyz_relative_error, "negative_samples": rec.negative_samples,
            "above_one_samples": int(np.count_nonzero(above)),
            "min": float(values.min()), "max": float(values.max()),
            "out_of_range": (values < -1e-12) | above,
            "reflected": tables.d65 * values if reflectance else None,
            "methods": {m: reconstruct(rec.target_rgb, reflectance=reflectance, method=m).values for m in METHODS},
            "matching": tables.xyz_matching, "weighted": a * values}


def spectrum_figure(key, data, details=None):
    details = details or spectrum_details(key, data)
    figure = Figure(figsize=(10, 6.5), dpi=110, layout="constrained")
    (ax, methods), (extra, cmf) = figure.subplots(2, 2, sharex=True)
    original, recovered = data["original"], data["reconstruction"].values
    wavelengths = np.asarray(WAVELENGTHS_NM)
    surface = key == "surface"
    ax.plot(wavelengths, original, label="Исходный")
    ax.plot(wavelengths, recovered, "--", label="Восстановленный")
    for bound in (0, 1) if surface else (0,):
        ax.axhline(bound, color="gray", linewidth=.8, linestyle=":")
    mask = details["out_of_range"]
    ax.scatter(wavelengths[mask], recovered[mask], color="#d62728", s=18, zorder=3, label="Вне допустимого диапазона")
    title = "Поверхность" if surface else f"Источник {key.split('_')[-1]}"
    ax.set(title=f"{title}: исходный и восстановленный", ylabel="Kd" if surface else "Спектр, 1/нм")
    lower = min(0, float(min(original.min(), recovered.min()))*1.05)
    upper = max(float(max(original.max(), recovered.max()))*1.05, 1.05 if surface else 1e-12)
    ax.set_ylim(lower, upper)
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    for name, label, style in (("basis", "Базисный", "-"), ("bounded", "С ограничениями", "--")):
        methods.plot(wavelengths, details["methods"][name], style, label=label)
    methods.axhline(0, color="gray", linewidth=.8, linestyle=":")
    methods.set(title="Сравнение методов для того же RGB")
    methods.legend(fontsize=8); methods.grid(alpha=.3)
    if surface:
        extra.plot(wavelengths, details["reflected"], color="#8c564b", label="D65·Kd")
        extra.plot(wavelengths, load_tables().d65, color="gray", linewidth=.8, label="D65")
        extra.set(title="Отражённый свет", ylabel="Отн. ед.")
        extra.legend(fontsize=8)
    else:
        difference = spectrum_display_delta(data)
        extra.plot(wavelengths, difference, color="#b44c40")
        if not np.any(difference):
            span = max(float(np.max(np.abs(original))) * .01, 1e-12)
            extra.set_ylim(-span, span)
        extra.set(title="Отклонение от исходного", ylabel="Отклонение")
    extra.axhline(0, color="gray", linewidth=.6)
    extra.set(xlabel="Длина волны, нм"); extra.grid(alpha=.3)
    xyz = details["actual_xyz"]
    # One shared scale keeps the relative X:Y:Z contributions visible on the CMF axis.
    scale = details["matching"].max() / max(float(np.abs(details["weighted"]).max()), 1e-300)
    for i, (name, color) in enumerate((("x̄", "#d62728"), ("ȳ", "#2ca02c"), ("z̄", "#1f77b4"))):
        cmf.plot(wavelengths, details["matching"][:, i], color=color, linewidth=.8, label=name)
        cmf.fill_between(wavelengths, details["weighted"][i] * scale, color=color, alpha=.18)
    cmf.axhline(0, color="gray", linewidth=.6)
    cmf.set(title=f"CMF, заливка - вклад отсчётов\nX={xyz[0]:.4g}, Y={xyz[1]:.4g}, Z={xyz[2]:.4g}",
            xlabel="Длина волны, нм")
    cmf.legend(fontsize=8); cmf.grid(alpha=.3)
    return figure


def brightness_rows(result):
    field = result.field
    rows = []
    for point in (*result.statistics.probes, result.statistics.minimum, result.statistics.maximum):
        ij = np.unravel_index(np.nanargmin(np.sum((field.surface.points_mm-np.asarray(point.point_mm))**2, axis=-1)), field.sphere_mask.shape)
        rows.append((point.title, *point.point_mm, point.luminance_w_m2_sr, 683*field.xyz_spectral[ij][1]))
    return rows


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
        writer.writerows(brightness_rows(result))
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
