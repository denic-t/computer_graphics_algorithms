"""Графический интерфейс ЛР3: таблицы спектров, два метода и сравнение."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from queue import SimpleQueue
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from .colorimetry import monitor_images
from .geometry import angles_to_target
from .models import LightSource, ParameterValidator, SceneParameters, Spectrum, WAVELENGTHS_NM
from .presets import gaussian_band, neutral_reflectance, white_d65
from .service import ComputationResult, LuminanceService

ROOT = Path(__file__).resolve().parent.parent
SCENE_FIELDS = (
    ("height_mm", "Высота экрана H, мм"),
    ("width_mm", "Ширина экрана W, мм"),
    ("height_px", "Разрешение Hres, пикс"),
    ("width_px", "Разрешение Wres, пикс"),
    ("observer_z_mm", "Наблюдатель zO, мм"),
    ("view_tilt_deg", "Наклон взгляда 0–180°, °"),
    ("view_azimuth_deg", "Азимут взгляда, °"),
    ("sphere_x_mm", "Центр сферы xC, мм"),
    ("sphere_y_mm", "Центр сферы yC, мм"),
    ("sphere_z_mm", "Центр сферы zC, мм"),
    ("sphere_radius_mm", "Радиус сферы R, мм"),
)
LIGHT_FIELDS = (
    ("x_mm", "xL, мм"), ("y_mm", "yL, мм"), ("z_mm", "zL, мм"),
    ("intensity_w_sr", "I0, Вт/ср"),
    ("axis_tilt_deg", "Наклон оси, °"),
    ("axis_azimuth_deg", "Азимут оси, °"),
)


def parse_number(raw: str, label: str, integer: bool = False) -> float | int:
    try:
        value = float(raw.strip().replace(",", "."))
        if integer:
            if not value.is_integer():
                raise ValueError
            return int(value)
        return value
    except ValueError as error:
        raise ValueError(f"{label}: введите {'целое ' if integer else ''}число.") from error


class SpectrumEditor(tk.Toplevel):
    """Готовые формы, полоса с управляемым пиком и точная таблица."""

    def __init__(
        self, master: tk.Misc, title: str, spectrum: Spectrum, save,
        *, maximum: float | None = None,
    ) -> None:
        super().__init__(master)
        self.title(title)
        self.geometry("640x700")
        self.minsize(520, 600)
        self.transient(master)
        self._spectrum_title = title
        self._maximum = maximum
        self._save_callback = save
        self._hint = tk.StringVar(value="Выберите образец или измените таблицу вручную.")

        presets = ttk.LabelFrame(self, text="Быстрый выбор спектра", padding=6)
        presets.pack(fill="x", padx=10, pady=(8, 4))
        if maximum is None:
            ttk.Button(presets, text="Белый D65", command=lambda: self._select_preset(white_d65())).grid(
                row=0, column=0, sticky="ew", padx=2, pady=2,
            )
        else:
            ttk.Button(presets, text="Нейтральный", command=lambda: self._select_preset(neutral_reflectance())).grid(
                row=0, column=0, sticky="ew", padx=2, pady=2,
            )
        for column, (label, peak, width) in enumerate(
            (("Красный", 630, 45), ("Зелёный", 540, 40),
             ("Синий", 460, 35), ("Жёлтый", 580, 40)), start=1,
        ):
            ttk.Button(
                presets, text=label,
                command=lambda p=peak, w=width: self._choose_band(p, w),
            ).grid(row=0, column=column, sticky="ew", padx=2, pady=2)
        for column in range(5):
            presets.columnconfigure(column, weight=1)

        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=10)
        self.peak_scale = tk.Scale(
            controls, from_=380, to=780, resolution=10, orient="horizontal",
            label="Пик, нм", length=240,
        )
        self.peak_scale.set(550)
        self.peak_scale.pack(side="left", fill="x", expand=True)
        self.width_scale = tk.Scale(
            controls, from_=10, to=150, resolution=5, orient="horizontal",
            label="Ширина полосы σ, нм", length=240,
        )
        self.width_scale.set(40)
        self.width_scale.pack(side="left", fill="x", expand=True)
        ttk.Button(self, text="Создать спектр по пику и ширине", command=self._use_band).pack(
            fill="x", padx=10, pady=(0, 4),
        )
        ttk.Label(
            self, text="Готовые варианты и полоса сразу применяются к сцене. "
                       "Для ручной таблицы нажмите «Применить таблицу». "
                       "Цветные образцы — учебные модели; D65 — стандартный спектр.",
            wraplength=600,
        ).pack(fill="x", padx=10)

        figure = Figure(figsize=(6, 1.6), dpi=100)
        self._axes = figure.add_subplot()
        self._plot = FigureCanvasTkAgg(figure, master=self)
        self._plot.get_tk_widget().pack(fill="x", padx=10, pady=4)

        ttk.Label(
            self, text="Точная таблица: 41 строка от 380 до 780 нм с шагом 10 нм. "
                 "Формат строки: 380; значение",
            wraplength=600,
        ).pack(fill="x", padx=10, pady=(4, 2))
        self.editor = scrolledtext.ScrolledText(self, font=("Consolas", 10), wrap="none")
        self.editor.pack(fill="both", expand=True, padx=10)
        ttk.Label(self, textvariable=self._hint).pack(fill="x", padx=10, pady=(3, 0))
        actions = ttk.Frame(self)
        actions.pack(fill="x", padx=10, pady=(2, 8))
        ttk.Button(actions, text="Показать график таблицы", command=self._preview_table).pack(
            side="left", fill="x", expand=True, padx=(0, 3),
        )
        ttk.Button(actions, text="Применить таблицу", command=self._apply).pack(
            side="left", fill="x", expand=True, padx=(3, 0),
        )
        self._replace_table(spectrum)
        self.grab_set()

    def _choose_band(self, peak_nm: int, width_nm: int) -> None:
        self.peak_scale.set(peak_nm)
        self.width_scale.set(width_nm)
        self._use_band()

    def _use_band(self) -> None:
        spectrum = gaussian_band(
            self.peak_scale.get(), self.width_scale.get(),
            reflectance=self._maximum is not None,
        )
        self._select_preset(spectrum)

    def _select_preset(self, spectrum: Spectrum) -> None:
        self._replace_table(spectrum)
        self._save_callback(spectrum)
        self._hint.set("Спектр применён к сцене; изображения пересчитаны.")

    def _replace_table(self, spectrum: Spectrum) -> None:
        self.editor.delete("1.0", "end")
        self.editor.insert(
            "1.0", "\n".join(
                f"{nm}; {value:.8g}" for nm, value in zip(WAVELENGTHS_NM, spectrum.values)
            ),
        )
        self._draw_preview(spectrum)
        self._hint.set("Спектр готов. Нажмите «Применить таблицу», чтобы изменить сцену.")

    def _draw_preview(self, spectrum: Spectrum) -> None:
        self._axes.clear()
        self._axes.plot(WAVELENGTHS_NM, spectrum.values, color="#326cb0")
        self._axes.set_xlim(380, 780)
        self._axes.set_ylim(bottom=0)
        self._axes.set_xlabel("Длина волны, нм")
        self._axes.set_ylabel("Отражение" if self._maximum is not None else "Форма спектра")
        self._axes.grid(True, alpha=0.3)
        self._axes.figure.tight_layout()
        self._plot.draw_idle()

    def _parse_table(self) -> Spectrum:
        rows = [line.strip() for line in self.editor.get("1.0", "end").splitlines() if line.strip()]
        if len(rows) != len(WAVELENGTHS_NM):
            raise ValueError("Нужно ровно 41 строку для 380–780 нм с шагом 10 нм.")
        values = []
        for expected, row in zip(WAVELENGTHS_NM, rows):
            parts = row.replace(";", " ").replace("\t", " ").split()
            if len(parts) != 2 or int(parts[0]) != expected:
                raise ValueError(f"Ожидалась строка для {expected} нм в формате «{expected}; значение».")
            values.append(float(parts[1].replace(",", ".")))
        spectrum = Spectrum.from_values(values)
        ParameterValidator._spectrum(spectrum, self._spectrum_title, maximum=self._maximum)
        return spectrum

    def _preview_table(self) -> None:
        try:
            spectrum = self._parse_table()
        except (ValueError, OverflowError) as error:
            self._hint.set(str(error))
            return
        self._draw_preview(spectrum)
        self._hint.set("График обновлён. Нажмите «Применить таблицу» для расчёта.")

    def _apply(self) -> None:
        try:
            spectrum = self._parse_table()
        except (ValueError, OverflowError) as error:
            messagebox.showerror("Ошибка таблицы", str(error), parent=self)
            return
        self._save_callback(spectrum)
        self.destroy()


class LightForm(ttk.LabelFrame):
    def __init__(self, master: tk.Misc, light: LightSource, number: int, changed, remove, spectrum_changed, aim) -> None:
        super().__init__(master, text=f"Источник {number}", padding=5)
        self.spectrum = light.spectrum
        self._spectrum_changed = spectrum_changed
        self.values: dict[str, tk.StringVar] = {}
        self._axis_azimuth_entry: ttk.Entry | None = None
        for row, (name, label) in enumerate(LIGHT_FIELDS):
            ttk.Label(self, text=label, width=17).grid(row=row, column=0, sticky="w")
            value = tk.StringVar(value=f"{getattr(light, name):g}")
            value.trace_add("write", lambda *_: changed())
            entry = ttk.Entry(self, textvariable=value, width=15)
            entry.grid(row=row, column=1, sticky="ew", pady=1)
            if name == "axis_azimuth_deg":
                self._axis_azimuth_entry = entry
            self.values[name] = value
        self.values["axis_tilt_deg"].trace_add("write", lambda *_: self._sync_axis_azimuth_state())
        self._sync_axis_azimuth_state()
        self.enabled = tk.BooleanVar(value=light.enabled)
        ttk.Checkbutton(self, text="Включён", variable=self.enabled, command=changed).grid(
            row=6, column=0, columnspan=2, sticky="w",
        )
        self.direction_status = tk.StringVar(value="Ось к центру сферы: —")
        ttk.Label(self, textvariable=self.direction_status).grid(
            row=7, column=0, columnspan=2, sticky="w",
        )
        ttk.Button(self, text="Направить на центр сферы", command=lambda: aim(self)).grid(
            row=8, column=0, columnspan=2, sticky="ew", pady=2,
        )
        ttk.Button(
            self, text="Изменить спектр источника…",
            command=lambda: SpectrumEditor(
                self, "Спектр источника", self.spectrum, self._set_spectrum,
            ),
        ).grid(row=9, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Button(self, text="Удалить", command=lambda: remove(self)).grid(
            row=10, column=0, columnspan=2, sticky="ew", pady=3,
        )

    def _sync_axis_azimuth_state(self) -> None:
        try:
            tilt = float(self.values["axis_tilt_deg"].get().replace(",", "."))
        except ValueError:
            tilt = None
        if self._axis_azimuth_entry is not None:
            self._axis_azimuth_entry.state(
                ["disabled"] if tilt in (0.0, 180.0) else ["!disabled"]
            )

    def _set_spectrum(self, spectrum: Spectrum) -> None:
        self.spectrum = spectrum
        self._spectrum_changed()

    def read(self) -> LightSource:
        numbers = {name: parse_number(value.get(), label) for (name, label), value in zip(LIGHT_FIELDS, self.values.values())}
        return LightSource(**numbers, spectrum=self.spectrum, enabled=self.enabled.get())


class SpectrumComparisonPanel(ttk.Frame):
    """Сравнение спектра выбранного источника с D65 без расчёта сцены."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, padding=8)
        self._lights: list[LightForm] = []
        self._selected_light: LightForm | None = None
        self._d65 = white_d65()

        controls = ttk.Frame(self)
        controls.pack(fill="x", pady=(0, 6))
        ttk.Label(controls, text="Источник света:").pack(side="left", padx=(0, 8))
        self.source = ttk.Combobox(controls, state="readonly", width=24)
        self.source.pack(side="left")
        self.source.bind("<<ComboboxSelected>>", self._source_selected)

        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.axes = self.figure.add_subplot()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        ttk.Button(self, text="Сохранить график спектров", command=self.save_graph).pack(
            anchor="e", pady=(6, 0),
        )
        self.refresh()

    def set_lights(self, lights: list[LightForm]) -> None:
        self._lights = list(lights)
        if self._selected_light not in self._lights:
            self._selected_light = self._lights[0] if self._lights else None
        self.source.configure(values=[str(light.cget("text")) for light in self._lights])
        if self._selected_light is None:
            self.source.set("")
        else:
            self.source.current(self._lights.index(self._selected_light))
        self.refresh()

    def _source_selected(self, _event: tk.Event) -> None:
        index = self.source.current()
        self._selected_light = self._lights[index] if 0 <= index < len(self._lights) else None
        self.refresh()

    def refresh(self) -> None:
        self.axes.clear()
        if self._selected_light is not None:
            selected = self._selected_light.spectrum.as_array()
            reference = self._d65.as_array()
            self.axes.plot(WAVELENGTHS_NM, selected / selected.max(), label="Выбранный источник")
            self.axes.plot(WAVELENGTHS_NM, reference / reference.max(), label="D65")
            self.axes.legend()
        self.axes.set(
            xlim=(380, 780), ylim=(0, 1.05),
            xlabel="Длина волны, нм", ylabel="Относительная интенсивность",
            title="Сравнение спектра источника с D65",
        )
        self.axes.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def save_graph(self) -> None:
        filename = filedialog.asksaveasfilename(
            parent=self, title="Сохранить график спектров", initialdir=str(ROOT),
            defaultextension=".png", filetypes=[("PNG", "*.png")],
        )
        if filename:
            self.figure.savefig(filename, format="png")


class ApplicationWindow(tk.Tk):
    """Окно с управлением сценой, спектрами, изображениями и значениями."""

    def __init__(self) -> None:
        super().__init__()
        self.title("ЛР №3 — моделирование цветных изображений")
        self.geometry("1480x900")
        self.minsize(1120, 700)
        self._result: ComputationResult | None = None
        self._job: str | None = None
        self._render_generation = 0
        self._render_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lab3-render")
        self._render_future: Future[ComputationResult] | None = None
        self._render_results: SimpleQueue[tuple[int, ComputationResult | ValueError]] = SimpleQueue()
        self._loading = False
        self._syncing = False
        self._photos: list[ImageTk.PhotoImage] = []
        self._lock_exposure = tk.BooleanVar(value=False)
        self._reference_y: float | None = None
        self._reflectance = SceneParameters.default().reflectance
        self._auto = tk.BooleanVar(value=True)
        self._square = tk.BooleanVar(value=True)
        self._full_visibility = tk.BooleanVar(value=False)
        self._status = tk.StringVar(value="Готово")

        ttk.Label(self, textvariable=self._status, anchor="w").pack(side="bottom", fill="x", padx=8, pady=3)
        split = ttk.PanedWindow(self, orient="horizontal")
        split.pack(fill="both", expand=True)
        left_container, left = self._scrolling_panel(split)
        split.add(left_container, weight=0)
        right = ttk.Frame(split, padding=8)
        split.add(right, weight=1)

        scene = ttk.LabelFrame(left, text="Экран, наблюдатель, сфера", padding=5)
        scene.pack(fill="x", pady=4)
        self._scene: dict[str, tk.StringVar] = {}
        self._scene_entries: dict[str, ttk.Entry] = {}
        for row, (name, label) in enumerate(SCENE_FIELDS):
            ttk.Label(scene, text=label, width=25).grid(row=row, column=0, sticky="w")
            value = tk.StringVar()
            value.trace_add("write", lambda *_ , key=name: self._scene_changed(key))
            entry = ttk.Entry(scene, textvariable=value, width=14)
            entry.grid(row=row, column=1, pady=1)
            self._scene[name] = value
            self._scene_entries[name] = entry
        ttk.Button(left, text="Направить взгляд на центр сферы", command=self._aim_view_at_sphere).pack(
            fill="x", pady=2,
        )
        ttk.Label(
            left, text="Азимут повторяется через 360°; при наклоне 0° или 180° он не влияет.",
            wraplength=330,
        ).pack(anchor="w", pady=2)
        ttk.Checkbutton(left, text="Подгонять квадратный пиксель", variable=self._square).pack(anchor="w")
        ttk.Checkbutton(
            left, text="Сфера целиком в кадре", variable=self._full_visibility,
            command=lambda: self.recalculate(interactive=False),
        ).pack(anchor="w", pady=2)
        ttk.Button(left, text="Таблица отражения сферы…", command=self._edit_reflectance).pack(fill="x", pady=4)

        self._lights_frame = ttk.LabelFrame(left, text="Источники света", padding=5)
        self._lights_frame.pack(fill="x", pady=4)
        ttk.Label(
            self._lights_frame,
            text="Наклон: 0° вниз (−Z), 90° горизонтально, 180° вверх. "
                 "Азимут: 0° к +X, 90° к +Y; повторяется через 360°.",
            wraplength=320,
        ).pack(anchor="w")
        self._light_forms: list[LightForm] = []
        ttk.Button(left, text="+ Источник", command=self._add_light).pack(fill="x", pady=2)
        ttk.Checkbutton(left, text="Автопересчёт", variable=self._auto).pack(anchor="w", pady=3)
        ttk.Button(left, text="Рассчитать (Enter)", command=self.recalculate).pack(fill="x", pady=2)
        ttk.Button(left, text="Сохранить 4 изображения и значения", command=self.save_current).pack(fill="x", pady=2)
        ttk.Button(left, text="Сравнить и сохранить белые/цветные источники", command=self.save_both).pack(fill="x", pady=2)
        ttk.Button(left, text="Значения по умолчанию", command=self.reset).pack(fill="x", pady=2)

        tabs = ttk.Notebook(right)
        self._tabs = tabs
        tabs.pack(fill="both", expand=True)
        images_tab = ttk.Frame(tabs)
        tabs.add(images_tab, text="Изображения")
        curves_tab = ttk.Frame(tabs)
        tabs.add(curves_tab, text="Сечения яркости")
        self._spectrum_comparison = SpectrumComparisonPanel(tabs)
        tabs.add(self._spectrum_comparison, text="Сравнение спектров")
        exposure_controls = ttk.Frame(images_tab)
        exposure_controls.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Checkbutton(
            exposure_controls, text="Зафиксировать экспозицию просмотра",
            variable=self._lock_exposure, command=self._toggle_exposure,
        ).pack(side="left", padx=5)
        ttk.Label(
            exposure_controls, text="Сохранённые PNG всегда нормируются по своему максимуму.",
        ).pack(side="left", padx=8)
        self._image_labels = []
        for row, title in enumerate(("image_1 — спектральная сцена", "image_2 — RGB-сцена", "image_1 — серый", "image_2 — серый")):
            frame = ttk.LabelFrame(images_tab, text=title, padding=3)
            frame.grid(row=row // 2 + 1, column=row % 2, sticky="nsew", padx=3, pady=3)
            label = ttk.Label(frame)
            label.pack(fill="both", expand=True)
            self._image_labels.append(label)
        for index in (0, 1):
            images_tab.columnconfigure(index, weight=1)
            images_tab.rowconfigure(index + 1, weight=1)
        self._figure = Figure(figsize=(8, 5), dpi=100)
        self._axes = self._figure.add_subplot()
        self._plot = FigureCanvasTkAgg(self._figure, master=curves_tab)
        self._plot.get_tk_widget().pack(fill="both", expand=True)
        report_frame = ttk.LabelFrame(right, text="Абсолютные значения и сравнение")
        report_frame.pack(fill="x", pady=5)
        self._report = tk.Text(report_frame, height=10, wrap="none", font=("Consolas", 9), state="disabled")
        self._report.pack(fill="x")
        self.bind("<Return>", self._on_return)
        self.bind("<<SceneComputed>>", self._receive_auto_result)
        self.reset()

    def _on_return(self, event: tk.Event) -> None:
        widget = event.widget
        while widget is not None:
            if widget is self._tabs:
                return
            widget = widget.master
        self.recalculate()

    def destroy(self) -> None:
        self._render_generation += 1
        self._render_executor.shutdown(wait=False, cancel_futures=True)
        super().destroy()

    @staticmethod
    def _scrolling_panel(split: ttk.PanedWindow) -> tuple[ttk.Frame, ttk.Frame]:
        container = ttk.Frame(split)
        canvas = tk.Canvas(container, width=365, highlightthickness=0)
        bar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        inner = ttk.Frame(canvas, padding=5)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        return container, inner

    def _scene_changed(self, key: str) -> None:
        if key == "view_tilt_deg":
            self._sync_view_azimuth_state()
        if self._loading or self._syncing:
            return
        if self._square.get() and key in ("height_mm", "width_mm", "height_px", "width_px"):
            try:
                values = {name: float(self._scene[name].get().replace(",", ".")) for name in ("height_mm", "width_mm", "height_px", "width_px")}
                h, w, hp, wp = (values[name] for name in ("height_mm", "width_mm", "height_px", "width_px"))
                if min(h, w, hp, wp) > 0:
                    self._syncing = True
                    if key == "height_mm": self._scene["width_mm"].set(f"{h * wp / hp:.6g}")
                    elif key == "width_mm": self._scene["height_mm"].set(f"{w * hp / wp:.6g}")
                    elif key == "height_px":
                        new_wp = max(1, round(hp * w / h))
                        self._scene["width_px"].set(str(new_wp))
                        self._scene["width_mm"].set(f"{new_wp * h / hp:.6g}")
                    else:
                        new_hp = max(1, round(wp * h / w))
                        self._scene["height_px"].set(str(new_hp))
                        self._scene["height_mm"].set(f"{new_hp * w / wp:.6g}")
            except ValueError:
                pass
            finally:
                self._syncing = False
        self._schedule()

    def _sync_view_azimuth_state(self) -> None:
        try:
            tilt = float(self._scene["view_tilt_deg"].get().replace(",", "."))
        except ValueError:
            tilt = None
        self._scene_entries["view_azimuth_deg"].state(
            ["disabled"] if tilt in (0.0, 180.0) else ["!disabled"]
        )

    def _aim_view_at_sphere(self) -> None:
        try:
            z_observer = parse_number(self._scene["observer_z_mm"].get(), "zO")
            center = tuple(
                parse_number(self._scene[key].get(), key)
                for key in ("sphere_x_mm", "sphere_y_mm", "sphere_z_mm")
            )
            tilt, azimuth = angles_to_target((0.0, 0.0, z_observer), center)
        except ValueError as error:
            messagebox.showerror("Направление взгляда", str(error))
            return
        was_loading = self._loading
        self._loading = True
        try:
            self._scene["view_tilt_deg"].set(f"{tilt:.6g}")
            self._scene["view_azimuth_deg"].set(f"{azimuth:.6g}")
        finally:
            self._loading = was_loading
        self.recalculate()

    def _aim_light_at_sphere(self, form: LightForm) -> None:
        try:
            origin = tuple(
                parse_number(form.values[key].get(), key) for key in ("x_mm", "y_mm", "z_mm")
            )
            center = tuple(
                parse_number(self._scene[key].get(), key)
                for key in ("sphere_x_mm", "sphere_y_mm", "sphere_z_mm")
            )
            tilt, azimuth = angles_to_target(origin, center)
        except ValueError as error:
            messagebox.showerror("Направление источника", str(error))
            return
        was_loading = self._loading
        self._loading = True
        try:
            form.values["axis_tilt_deg"].set(f"{tilt:.6g}")
            form.values["axis_azimuth_deg"].set(f"{azimuth:.6g}")
        finally:
            self._loading = was_loading
        self.recalculate()

    def _schedule(self) -> None:
        if self._loading or not self._auto.get():
            return
        self._render_generation += 1
        if self._job is not None:
            self.after_cancel(self._job)
        if self._render_future is not None:
            self._render_future.cancel()
        self._job = self.after(650, self._start_auto_calculation)

    def _start_auto_calculation(self) -> None:
        self._job = None
        if not self._auto.get():
            return
        try:
            parameters = self.read_parameters()
        except ValueError as error:
            self._clear_result(error)
            self._status.set(f"⚠ {error}")
            return
        generation = self._render_generation
        future = self._render_executor.submit(LuminanceService.compute, parameters)
        self._render_future = future

        def completed(done: Future[ComputationResult]) -> None:
            if done.cancelled():
                return
            try:
                outcome: ComputationResult | ValueError = done.result()
            except ValueError as error:
                outcome = error
            self._render_results.put((generation, outcome))
            try:
                self.event_generate("<<SceneComputed>>", when="tail")
            except (RuntimeError, tk.TclError):
                pass  # Окно могло быть закрыто во время расчёта.

        future.add_done_callback(completed)

    def _receive_auto_result(self, _event: tk.Event) -> None:
        while not self._render_results.empty():
            generation, outcome = self._render_results.get_nowait()
            if generation != self._render_generation:
                continue
            self._render_future = None
            if isinstance(outcome, ValueError):
                self._clear_result(outcome)
                self._status.set(f"⚠ {outcome}")
            else:
                self._show_result(outcome)

    def _edit_reflectance(self) -> None:
        SpectrumEditor(
            self, "Спектр отражения сферы", self._reflectance,
            self._set_reflectance, maximum=1,
        )

    def _set_reflectance(self, spectrum: Spectrum) -> None:
        self._reflectance = spectrum
        self._spectrum_changed()

    def _spectrum_changed(self) -> None:
        self._spectrum_comparison.refresh()
        if self.recalculate(interactive=False):
            self._status.set("Спектр применён. Изображения пересчитаны.")

    def _add_light(self, light: LightSource | None = None) -> None:
        if len(self._light_forms) >= 6:
            messagebox.showerror("Источники", "Допускается не более 6 источников.")
            return
        if light is None:
            white = SceneParameters.default().with_white_lights().lights[0]
            light = replace(white, x_mm=0, y_mm=0, z_mm=2000, axis_tilt_deg=0, axis_azimuth_deg=0)
        form = LightForm(
            self._lights_frame, light, len(self._light_forms) + 1,
            self._schedule, self._remove_light, self._spectrum_changed,
            self._aim_light_at_sphere,
        )
        form.pack(fill="x", pady=3)
        self._light_forms.append(form)
        self._spectrum_comparison.set_lights(self._light_forms)
        self._schedule()

    def _remove_light(self, form: LightForm) -> None:
        if len(self._light_forms) == 1:
            messagebox.showerror("Источники", "Нужен хотя бы один источник.")
            return
        self._light_forms.remove(form)
        form.destroy()
        for number, item in enumerate(self._light_forms, 1):
            item.configure(text=f"Источник {number}")
        self._spectrum_comparison.set_lights(self._light_forms)
        self._schedule()

    def read_parameters(self) -> SceneParameters:
        values = {name: parse_number(self._scene[name].get(), label, name in ("height_px", "width_px")) for name, label in SCENE_FIELDS}
        return SceneParameters(
            **values, reflectance=self._reflectance,
            lights=tuple(form.read() for form in self._light_forms),
            require_full_visibility=self._full_visibility.get(),
        )

    @staticmethod
    def _maximum_y(result: ComputationResult) -> float:
        return max(
            float(np.max(result.field.xyz_spectral[..., 1])),
            float(np.max(result.field.xyz_rgb[..., 1])),
        )

    def _toggle_exposure(self) -> None:
        self._reference_y = self._maximum_y(self._result) if self._lock_exposure.get() and self._result else None
        self._redraw_images()

    def _redraw_images(self) -> None:
        if self._result is None:
            return
        result = self._result
        if self._lock_exposure.get():
            if self._reference_y is None:
                self._reference_y = self._maximum_y(result)
            images = monitor_images(
                result.field.xyz_spectral, result.field.xyz_rgb,
                reference_y=self._reference_y,
            )
        else:
            images = result.images
        self._photos.clear()
        for pixels, label in zip(images, self._image_labels):
            image = Image.fromarray(pixels)
            image.thumbnail((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self._photos.append(photo)
            label.configure(image=photo, text="")

    def _update_light_directions(self, parameters: SceneParameters) -> None:
        center = np.asarray(parameters.sphere_center_mm, dtype=float)
        for form, light in zip(self._light_forms, parameters.lights):
            toward_center = center - np.asarray(light.position_mm, dtype=float)
            toward_center /= np.linalg.norm(toward_center)
            cosine = float(np.clip(light.axis @ toward_center, -1, 1))
            angle = float(np.degrees(np.arccos(cosine)))
            suffix = " (выключен)" if not light.enabled else ""
            form.direction_status.set(
                f"Ось к центру: {angle:.1f}°, cosθ = {max(0, cosine):.2f}{suffix}"
            )

    def _clear_result(self, error: ValueError) -> None:
        self._result = None
        self._photos.clear()
        for label in self._image_labels:
            label.configure(image="", text="Нет изображения: исправьте параметры сцены")
        self._axes.clear()
        self._plot.draw_idle()
        self._report.configure(state="normal")
        self._report.delete("1.0", "end")
        self._report.insert("1.0", str(error))
        self._report.configure(state="disabled")

    def recalculate(self, *, interactive: bool = True) -> bool:
        self._render_generation += 1
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None
        if self._render_future is not None:
            self._render_future.cancel()
        try:
            parameters = self.read_parameters()
            result = LuminanceService.compute(parameters)
        except ValueError as error:
            self._clear_result(error)
            self._status.set(f"⚠ {error}")
            if interactive: messagebox.showerror("Ошибка параметров", str(error))
            return False
        self._show_result(result)
        return True

    def _show_result(self, result: ComputationResult) -> None:
        parameters = result.parameters
        self._result = result
        self._redraw_images()
        self._update_light_directions(parameters)
        self._axes.clear()
        for section in result.cross_sections:
            self._axes.plot(section.coordinates_mm, section.values_w_m2_sr, label=section.title)
        self._axes.set(xlabel="Координата на экране, мм", ylabel="Радианс, Вт/(м²·ср)", title="Сечения видимой части сферы")
        self._axes.grid(True, linestyle=":")
        self._axes.legend()
        self._figure.tight_layout()
        self._plot.draw_idle()
        self._report.configure(state="normal")
        self._report.delete("1.0", "end")
        self._report.insert("1.0", result.report)
        self._report.configure(state="disabled")
        self._status.set(
            f"Расчёт выполнен. Lср = {result.statistics.mean_w_m2_sr:.3e} Вт/(м²·ср); "
            f"отличие image_1/image_2 = {result.mean_difference:.2f} уровня."
        )

    def save_current(self) -> None:
        if not self.recalculate() or self._result is None:
            return
        directory = filedialog.askdirectory(title="Каталог для 4 изображений и отчёта", initialdir=str(ROOT))
        if directory:
            files = LuminanceService.save(self._result, Path(directory))
            self._status.set(f"Сохранено {len(files)} файлов в {directory}")

    def save_both(self) -> None:
        try:
            parameters = self.read_parameters()
            results = LuminanceService.compare_white_and_colored(parameters)
        except ValueError as error:
            messagebox.showerror("Ошибка параметров", str(error))
            return
        directory = filedialog.askdirectory(title="Каталог сравнения белых и цветных источников", initialdir=str(ROOT))
        if not directory:
            return
        files = LuminanceService.save_comparison(results, Path(directory))
        self._status.set(f"Сохранено {len(files)} файлов в {directory}")
        messagebox.showinfo("Сравнение готово", "Белые источники: среднее отличие {:.2f}\nЦветные источники: среднее отличие {:.2f}".format(
            results["white"].mean_difference, results["colored"].mean_difference))

    def reset(self) -> None:
        defaults = SceneParameters.default()
        self._reference_y = None
        self._loading = True
        for name, _ in SCENE_FIELDS:
            self._scene[name].set(f"{getattr(defaults, name):g}")
        self._full_visibility.set(defaults.require_full_visibility)
        self._sync_view_azimuth_state()
        self._reflectance = defaults.reflectance
        for form in self._light_forms:
            form.destroy()
        self._light_forms.clear()
        for light in defaults.lights:
            self._add_light(light)
        self._loading = False
        self.recalculate(interactive=False)
