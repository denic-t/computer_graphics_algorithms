"""Проверки числовых полей и применения спектральных таблиц в Tk."""

import tkinter as tk
from tkinter import ttk

import numpy as np
import pytest

from sphere_color.models import Spectrum, WAVELENGTHS_NM
from sphere_color.presets import gaussian_band
from sphere_color.ui import ApplicationWindow, SpectrumEditor, parse_number


def test_parse_number_accepts_decimal_comma_and_integer_resolution() -> None:
    assert parse_number("1,25", "значение") == 1.25
    assert parse_number("400", "разрешение", integer=True) == 400
    with pytest.raises(ValueError):
        parse_number("400,5", "разрешение", integer=True)


def test_spectrum_editor_applies_source_and_validates_reflectance(monkeypatch) -> None:
    """Повторяет пользовательское нажатие «Применить таблицу»."""
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Графический дисплей Tk недоступен")
    root.withdraw()
    try:
        saved = []
        rows = "\n".join(f"{nm}; 1.2" for nm in WAVELENGTHS_NM)
        source = SpectrumEditor(root, "Спектр источника", Spectrum.from_values([0.5] * 41), saved.append)
        source.editor.delete("1.0", "end")
        source.editor.insert("1.0", rows)
        source._apply()
        assert len(saved) == 1
        assert saved[0].values == (1.2,) * 41

        errors = []
        monkeypatch.setattr("sphere_color.ui.messagebox.showerror", lambda *args, **kwargs: errors.append(args))
        reflectance = SpectrumEditor(
            root, "Спектр отражения сферы", Spectrum.from_values([0.5] * 41),
            saved.append, maximum=1,
        )
        reflectance.editor.delete("1.0", "end")
        reflectance.editor.insert("1.0", rows)
        reflectance._apply()
        assert errors
        assert len(saved) == 1
        reflectance.destroy()
    finally:
        root.destroy()


def test_blue_presets_recalculate_both_sources_without_extra_apply() -> None:
    try:
        app = ApplicationWindow()
    except tk.TclError:
        pytest.skip("Графический дисплей Tk недоступен")
    app.withdraw()
    try:
        before = app._result.image_1.copy()
        app._auto.set(False)
        for source in app._light_forms:
            source.grid_slaves(row=9, column=0)[0].invoke()
            editor = next(child for child in source.winfo_children() if isinstance(child, SpectrumEditor))
            presets = next(child for child in editor.winfo_children() if isinstance(child, ttk.LabelFrame))
            blue = next(child for child in presets.winfo_children() if child.cget("text") == "Синий")
            blue.invoke()
            editor.destroy()
        for light in app.read_parameters().lights:
            assert light.spectrum.values == pytest.approx(gaussian_band(460, 35).values, rel=1e-7)
        difference = np.abs(before.astype(np.int16) - app._result.image_1.astype(np.int16))
        assert difference[app._result.field.sphere_mask].mean() > 10
        assert "пересчитаны" in app._status.get()
    finally:
        app.destroy()


def test_light_angles_change_radiance_while_exposure_reference_stays_fixed() -> None:
    try:
        app = ApplicationWindow()
    except tk.TclError:
        pytest.skip("Графический дисплей Tk недоступен")
    app.withdraw()
    try:
        initial_mean = app._result.statistics.mean_w_m2_sr
        app._auto.set(False)
        app._lock_exposure.set(True)
        app._toggle_exposure()
        reference = app._reference_y
        source = app._light_forms[0]
        assert "cosθ = 1.00" in source.direction_status.get()
        source.values["axis_azimuth_deg"].set("90")
        assert app.recalculate()
        assert app._reference_y == reference
        assert app._result.statistics.mean_w_m2_sr < initial_mean
        assert "cosθ = 0.46" in source.direction_status.get()
    finally:
        app.destroy()


def test_full_visibility_toggle_and_aim_controls() -> None:
    try:
        app = ApplicationWindow()
    except tk.TclError:
        pytest.skip("Графический дисплей Tk недоступен")
    app.withdraw()
    try:
        assert not app._full_visibility.get()
        assert app._scene_entries["view_azimuth_deg"].instate(["disabled"])
        app._auto.set(False)
        app._scene["sphere_x_mm"].set("900")
        assert app.recalculate()
        app._full_visibility.set(True)
        assert not app.recalculate(interactive=False)
        assert "целиком" in app._status.get()
        app._full_visibility.set(False)
        assert app.recalculate()

        app._scene["sphere_x_mm"].set("0")
        app._scene["view_tilt_deg"].set("25")
        assert app._scene_entries["view_azimuth_deg"].instate(["!disabled"])
        app._aim_view_at_sphere()
        assert float(app._scene["view_tilt_deg"].get()) == pytest.approx(0)
        assert app._scene_entries["view_azimuth_deg"].instate(["disabled"])

        source = app._light_forms[0]
        source.values["axis_tilt_deg"].set("0")
        assert source._axis_azimuth_entry.instate(["disabled"])
        source.values["axis_tilt_deg"].set("90")
        assert source._axis_azimuth_entry.instate(["!disabled"])
        source.grid_slaves(row=8, column=0)[0].invoke()
        assert "cosθ = 1.00" in source.direction_status.get()
    finally:
        app.destroy()
