"""Acceptance tests: color preservation, physical units, comparisons and replay."""
from dataclasses import replace
import json
import math
import time
import tkinter as tk
import numpy as np
import pytest
from sphere_color.colorimetry import spectrum_to_xyz, xyz_to_rgb, reflectance_to_rgb
from sphere_color.models import LightSource, Spectrum
from sphere_color.lab4_service import Lab4Service, default_scene, display_images, split_image
from sphere_color.reconstruction import reconstruct
from sphere_color.exporting import read_scene


@pytest.fixture(scope="module")
def result():
    return Lab4Service.compute(replace(default_scene(),height_px=200,width_px=200))


@pytest.mark.parametrize("reflectance", [False, True])
def test_lecture_basis_preserves_arbitrary_valid_color(reflectance):
    spectrum = .25+.6*np.exp(-.5*((np.arange(380,781,10)-520)/60)**2)
    rgb = reflectance_to_rgb(spectrum) if reflectance else xyz_to_rgb(spectrum_to_xyz(spectrum))
    recovered = reconstruct(rgb, reflectance=reflectance, method="basis")
    np.testing.assert_allclose(recovered.actual_rgb,rgb,rtol=1e-10,atol=1e-10)
    assert recovered.xyz_relative_error < 1e-12


def test_bounded_reflectance_reports_infeasible_color():
    rec=reconstruct([2,2,2],reflectance=True)
    assert np.all((rec.values>=0)&(rec.values<=1))
    assert rec.xyz_relative_error > .4


def test_source_power_is_not_hidden_by_reconstruction_or_monitor_normalization(result):
    p=result.parameters
    twice=replace(p,lights=tuple(replace(v,intensity_w_sr=2*v.intensity_w_sr) for v in p.lights))
    doubled=Lab4Service.compute(twice)
    np.testing.assert_allclose(doubled.field.values_w_m2_sr,2*result.field.values_w_m2_sr,rtol=1e-12)
    np.testing.assert_array_equal(doubled.image_A,result.image_A)
    fixed=display_images(doubled.xyz_fields,reference_y=max(v[...,1].max() for v in result.xyz_fields))
    assert fixed[0].mean()>result.image_A.mean()


def test_absolute_radiance_matches_independent_lambert_formula():
    p=default_scene()
    p=replace(p,height_px=200,width_px=200,lights=(LightSource(0,0,2000,100,Spectrum.from_values([1]*41)),))
    r=Lab4Service.compute(p)
    positions=r.field.surface.points_mm[r.field.sphere_mask]/1000
    normals=r.field.surface.normals[r.field.sphere_mask]
    ray=positions-np.array([0,0,2.])
    d2=(ray*ray).sum(axis=1)
    unit=ray/np.sqrt(d2)[:,None]
    expected=100*.7*np.maximum(-unit[:,2],0)*np.maximum(-(unit*normals).sum(axis=1),0)/(math.pi*d2)
    # Flat emitter is outside the D65 polynomial subspace; energy need not be preserved.
    # A source of D65 is in the basis and must match the physical reference exactly.
    from sphere_color.colorimetry import load_tables
    p=replace(p,lights=(replace(p.lights[0],spectrum=Spectrum.from_values(load_tables().d65)),))
    r=Lab4Service.compute(p)
    np.testing.assert_allclose(r.field.values_w_m2_sr[r.field.sphere_mask],expected,rtol=2e-6,atol=1e-8)


def test_white_neutral_three_paths_agree(result):
    np.testing.assert_array_equal(result.image_A,result.image_1)
    np.testing.assert_array_equal(result.image_A,result.image_2)
    np.testing.assert_allclose(result.field.values_w_m2_sr,result.original_field.values_w_m2_sr,rtol=2e-6,atol=1e-8)
    assert len(result.statistics.probes)==3
    assert result.statistics.minimum.luminance_w_m2_sr<=result.statistics.maximum.luminance_w_m2_sr


def test_split_view_edges_and_interior(result):
    left=np.full_like(result.image_A,17); right=np.full_like(left,53)
    assert np.all(split_image(left,right,0)==53)
    assert np.all(split_image(left,right,1)==17)
    image=split_image(left,right,.37); x=round(left.shape[1]*.37)
    assert np.all(image[:,:x]==17) and np.all(image[:,x:]==53)


def test_export_roundtrip_preserves_numeric_arrays_and_rgb_overrides(result,tmp_path):
    from PIL import Image
    files=Lab4Service.save(result,tmp_path,fraction=.37)
    assert all(p.is_file() and p.stat().st_size>0 for p in files)
    p,method,overrides=read_scene(tmp_path/'scene.json')
    replay=Lab4Service.compute(p,method,overrides)
    np.testing.assert_array_equal(replay.image_A,result.image_A)
    np.testing.assert_allclose(replay.field.xyz_spectral,result.field.xyz_spectral,atol=1e-12)
    with np.load(tmp_path/'absolute_fields.npz') as fields:
        assert fields['spectral_A_W_m2_sr_nm'].shape==(200,200,41)
        np.testing.assert_allclose(fields['xyz_A'],result.field.xyz_spectral)
    assert np.asarray(Image.open(tmp_path/'split_A_vs_1.png')).shape==(200,200,3)
    import csv
    with (tmp_path/'spectrum_surface.csv').open(encoding='utf-8-sig') as f:
        assert len(list(csv.DictReader(f)))==41


def test_invalid_inputs():
    for rgb in ([float('nan'),0,0],[-100,0,0],[1,2]):
        with pytest.raises(ValueError): reconstruct(rgb)
    with pytest.raises(ValueError): Lab4Service.compute(replace(default_scene(),width_px=199))


def test_unlit_sphere_returns_black_instead_of_crashing():
    p=default_scene()
    p=replace(p,height_px=200,width_px=200,lights=tuple(replace(v,axis_tilt_deg=180) for v in p.lights))
    r=Lab4Service.compute(p)
    assert not np.any(r.image_A)
    assert r.statistics.maximum.luminance_w_m2_sr==0


def test_identical_and_disabled_lights_have_distinct_spectrum_keys():
    p=default_scene(); light=p.lights[0]
    p=replace(p,height_px=200,width_px=200,lights=(replace(light,enabled=False),light,light))
    r=Lab4Service.compute(p)
    assert set(r.spectra)=={'surface','light_2','light_3'}


def test_real_tk_events_autorecalculate_and_split_drag():
    from sphere_color.lab4_ui import ApplicationWindow
    try: app=ApplicationWindow()
    except tk.TclError: pytest.skip('Tk display unavailable')
    app.withdraw()
    try:
        app._scene['height_px'].set('200')
        app._scene['width_px'].set('200')
        assert app.recalculate(interactive=False)
        app.update()
        before=app._result.statistics.mean_w_m2_sr
        app._light_forms[0].values['intensity_w_sr'].set('1500')
        deadline=time.monotonic()+8
        while time.monotonic()<deadline and app._result.statistics.mean_w_m2_sr==before:
            app.update(); time.sleep(.02)
        assert app._result.statistics.mean_w_m2_sr>before
        assert app._job is None
        assert len(app._table.get_children())==41
        assert app.title() == 'ЛР №4'
        assert set(app._entity['values'])=={'Поверхность','Источник 1','Источник 2'}
        assert 'ЛР3' not in app._report.get('1.0', 'end')
        app._entity_label.set('Источник 1'); app._choose_entity()
        assert app._selected.get() == 'light_1'
        app._pair_label.set('RGB-расчёт'); app._choose_pair()
        assert app._pair.get() == 'image_2'
        app._method_label.set('Базисный'); app._choose_method()
        assert app._result.method == 'basis'
        assert [app._tabs.tab(tab, 'text') for tab in app._tabs.tabs()] == ['Сцена', 'Сравнение', 'Спектры', 'Яркость']
        assert len(app._image_labels) == 3
        assert not hasattr(app, '_pixel')
        app._scene['height_mm'].set('800')
        assert float(app._scene['width_mm'].get())==800
        assert app.recalculate(interactive=False)
        x,w=app._split_bounds
        from types import SimpleNamespace
        app._drag(SimpleNamespace(x=x+w*.75))
        assert app._fraction.get()==pytest.approx(.75)
        app._pair.set('image_2'); app._draw_split()
        app._scene['sphere_radius_mm'].set('invalid')
        assert not app.recalculate(interactive=False)
        assert app._result is None
        assert not app._table.get_children()
    finally: app.destroy()
