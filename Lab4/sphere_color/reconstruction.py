"""RGB → XYZ → 41 samples. Lecture 4, slides 37–42; no source spectrum in solver."""
from dataclasses import dataclass
import numpy as np
from .colorimetry import load_tables, rgb_to_xyz, spectrum_to_xyz, xyz_to_rgb

METHODS = ("bounded", "basis")


def integration_matrix():
    weights = np.full(41, 10.0)
    weights[[0, -1]] = 5.0
    return load_tables().xyz_matching.T * weights


@dataclass(frozen=True)
class Reconstruction:
    values: np.ndarray
    target_rgb: np.ndarray
    actual_rgb: np.ndarray
    xyz_relative_error: float
    negative_samples: int
    coefficients: np.ndarray


def reconstruct(rgb, *, reflectance=False, method="bounded"):
    """CIE RGB, never 8-bit sRGB. Reflectance RGB is relative to D65.

    The basis spans D65*(1,t,t²) for emitters and (1,t,t²) for reflectors.
    bounded projects the exact basis solution onto physical sample bounds,
    then minimizes color error and a weak first-derivative smoothness penalty.
    It can have a nonzero color residual, which is returned explicitly.
    """
    rgb = np.asarray(rgb, dtype=float)
    if rgb.shape != (3,) or not np.isfinite(rgb).all():
        raise ValueError("RGB: нужны три конечных числа CIE RGB.")
    if method not in METHODS:
        raise ValueError("Неизвестный метод восстановления.")
    a = integration_matrix()
    d65 = load_tables().d65
    if reflectance:
        white_rgb = xyz_to_rgb(spectrum_to_xyz(d65))
        target = rgb_to_xyz(rgb * white_rgb)
        a = a * d65
    else:
        target = rgb_to_xyz(rgb)
    if np.any(target < -1e-10):
        raise ValueError("Этот CIE RGB задаёт отрицательный XYZ; физический цвет невозможен.")
    t = np.linspace(-1, 1, 41)
    basis = np.stack((np.ones(41), t, t*t), axis=1)
    if not reflectance:
        basis = basis * d65[:, None]
    coefficients = np.linalg.solve(a @ basis, target)
    values = basis @ coefficients
    upper_bound = 1.0 if reflectance else np.inf
    if method == "bounded" and (np.any(values < 0) or np.any(values > upper_bound)):
        # Normalize variables so the conditioning is independent of source power.
        scale = 1.0 if reflectance else max(float(np.max(np.abs(values))), 1e-12)
        an = a / np.linalg.norm(a, 2)
        bn = target / (scale * np.linalg.norm(a, 2))
        derivative = np.diff(np.eye(41), axis=0)
        h = an.T @ an + 1e-8 * (derivative.T @ derivative)
        c = an.T @ bn
        step = 1 / np.linalg.eigvalsh(h)[-1]
        upper = 1.0 if reflectance else np.inf
        x = np.clip(values / scale, 0, upper)
        z, momentum = x.copy(), 1.0
        for _ in range(2500):
            new = np.clip(z - step * (h @ z - c), 0, upper)
            next_momentum = (1 + np.sqrt(1 + 4*momentum*momentum)) / 2
            z = new + (momentum-1)/next_momentum * (new-x)
            x, momentum = new, next_momentum
        values = x * scale
    actual_xyz = a @ values
    actual_rgb = xyz_to_rgb(actual_xyz)
    if reflectance:
        actual_rgb = actual_rgb / white_rgb
    return Reconstruction(values, rgb, actual_rgb,
        float(np.linalg.norm(actual_xyz-target) / max(np.linalg.norm(target), 1e-15)),
        int(np.count_nonzero(values < -1e-12)), coefficients)
