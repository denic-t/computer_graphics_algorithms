"""Lab4 extends the original Lab3 controls; all updates are event-triggered."""
from dataclasses import replace
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .ui import ApplicationWindow as Lab3Window, SCENE_FIELDS
from .models import SceneParameters
from .lab4_service import Lab4Service, default_scene, display_images, split_image
from .exporting import spectrum_figure, read_scene, spectrum_display_delta, spectrum_details

METHOD_LABELS = {"bounded": "С ограничениями", "basis": "Базисный"}
IMAGE_LABELS = {"image_1": "Исходный спектр", "image_2": "RGB-расчёт"}


def entity_label(key):
    return "Поверхность" if key == "surface" else f"Источник {key.split('_')[-1]}"


class ApplicationWindow(Lab3Window):
    def __init__(self):
        self._method_name = "bounded"
        self._rgb_overrides = {}
        self._details = None
        self._lab4_ready = False
        super().__init__()
        self.title("ЛР №4")
        self._method = tk.StringVar(value="bounded")
        self._pair = tk.StringVar(value="image_1")
        self._fraction = tk.DoubleVar(value=.5)
        self._selected = tk.StringVar(value="surface")
        self._tabs.tab(0, text="Сцена")
        for label, title in zip(self._image_labels, ("Восстановленный спектр", "Исходный спектр", "RGB-расчёт", "Яркость")):
            label.master.configure(text=title)
        self._image_labels[-1].master.grid_remove()
        self._image_labels = self._image_labels[:3]
        image_tab = self._image_labels[0].master.master
        for column, label in enumerate(self._image_labels):
            label.master.grid(row=1, column=column, sticky="nsew", padx=3, pady=3)
            image_tab.columnconfigure(column, weight=1)
        image_tab.rowconfigure(2, weight=0)
        # Replace legacy export buttons without changing the Lab3 implementation.
        for child in self._lights_frame.master.winfo_children():
            if isinstance(child, ttk.Button):
                if str(child.cget("text")).startswith("Сохранить 4"):
                    child.configure(text="Сохранить результаты…", command=self.save_current)
                elif str(child.cget("text")).startswith("Сравнить и сохранить"):
                    child.configure(text="Открыть сцену…", command=self.load_scene)
        spectra_tab = ttk.Frame(self._tabs, padding=5)
        self._tabs.add(spectra_tab, text="Спектры")
        controls = ttk.Frame(spectra_tab); controls.pack(fill="x")
        ttk.Label(controls, text="Метод:").pack(side="left")
        self._method_label = tk.StringVar(value=METHOD_LABELS[self._method.get()])
        method = ttk.Combobox(controls, textvariable=self._method_label, values=tuple(METHOD_LABELS.values()), width=20, state="readonly")
        method.pack(side="left", padx=5)
        method.bind("<<ComboboxSelected>>", self._choose_method)
        ttk.Label(controls, text="Объект:").pack(side="left", padx=(12, 0))
        self._entity_label = tk.StringVar(value="Поверхность")
        self._entity = ttk.Combobox(controls, textvariable=self._entity_label, state="readonly", width=16)
        self._entity.pack(side="left", padx=5)
        self._entity.bind("<<ComboboxSelected>>", self._choose_entity)
        rgb_row = ttk.Frame(spectra_tab); rgb_row.pack(fill="x", pady=4)
        ttk.Label(rgb_row, text="CIE RGB:").pack(side="left")
        self._rgb_vars = [tk.StringVar() for _ in range(3)]
        for name, v in zip(("R", "G", "B"), self._rgb_vars):
            ttk.Label(rgb_row, text=name).pack(side="left", padx=(8, 2))
            ttk.Entry(rgb_row, textvariable=v, width=12).pack(side="left", padx=2)
        ttk.Button(rgb_row, text="Применить RGB", command=self._apply_rgb).pack(side="left", padx=3)
        ttk.Button(spectra_tab, text="Сбросить цвет", command=self._reset_rgb).pack(anchor="w")
        self._spectrum_info = tk.StringVar()
        ttk.Label(spectra_tab, textvariable=self._spectrum_info, wraplength=900).pack(fill="x", pady=4)
        self._calculation_info = tk.StringVar()
        ttk.Label(spectra_tab, textvariable=self._calculation_info, wraplength=1100, font=("Consolas", 9)).pack(fill="x", pady=(0, 4))
        panes = ttk.PanedWindow(spectra_tab, orient="horizontal"); panes.pack(fill="both", expand=True)
        table_frame = ttk.Frame(panes); panes.add(table_frame, weight=1)
        self._table = ttk.Treeview(table_frame, columns=("nm", "original", "restored", "delta"), show="headings", height=15)
        for key,title in zip(self._table["columns"], ("нм", "Исходный", "Восст.", "Разность")):
            self._table.heading(key, text=title); self._table.column(key, width=90, minwidth=45, stretch=True)
        self._table.bind("<Configure>", lambda event: [self._table.column(key, width=max(45,(event.width-8)//4)) for key in self._table["columns"]])
        scrollbar = ttk.Scrollbar(table_frame, command=self._table.yview)
        self._table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y"); self._table.pack(fill="both", expand=True)
        self._spectrum_frame = ttk.Frame(panes); panes.add(self._spectrum_frame, weight=2)
        self._spectrum_plot = None
        comparison = ttk.Frame(self._tabs, padding=8)
        self._tabs.add(comparison, text="Сравнение")
        pair_controls = ttk.Frame(comparison); pair_controls.pack(fill="x")
        ttk.Label(pair_controls, text="Сравнить с:").pack(side="left")
        self._pair_label = tk.StringVar(value=IMAGE_LABELS[self._pair.get()])
        pair = ttk.Combobox(pair_controls, textvariable=self._pair_label, values=tuple(IMAGE_LABELS.values()), state="readonly", width=20)
        pair.pack(side="left", padx=5)
        pair.bind("<<ComboboxSelected>>", self._choose_pair)
        ttk.Scale(pair_controls, from_=0, to=1, variable=self._fraction, command=lambda _: self._draw_split()).pack(side="left", fill="x", expand=True)
        self._comparison_info = tk.StringVar()
        ttk.Label(comparison, textvariable=self._comparison_info).pack(fill="x", pady=6)
        self._split_canvas = tk.Canvas(comparison, background="#18202b", highlightthickness=0)
        self._split_canvas.pack(fill="both", expand=True)
        self._split_canvas.bind("<Configure>", lambda _: self._draw_split())
        self._split_canvas.bind("<Button-1>", self._drag)
        self._split_canvas.bind("<B1-Motion>", self._drag)
        self._split_bounds = (0, 1)
        self._lab4_ready = True
        self._refresh_lab4()
        self._square.set(True)
        ttk.Button(self._lights_frame.master, text="Цветная сфера", command=self.load_lab3_white).pack(fill="x", pady=4)
        self._tabs.insert(1, comparison)
        self._tabs.insert(2, spectra_tab)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    @staticmethod
    def default_scene():
        return default_scene()

    def _compute(self, parameters):
        method = self._method.get() if hasattr(self, "_method") else self._method_name
        return Lab4Service.compute(parameters, method, self._rgb_overrides)

    def recalculate(self, *, interactive=True):
        success = super().recalculate(interactive=interactive)
        if success:
            self._status.set("Готово")
            self._show_summary()
        if self._lab4_ready:
            self._refresh_lab4()
        return success

    @staticmethod
    def _maximum_y(result):
        return max(max(float(v[..., 1].max()) for v in result.xyz_fields), 1e-15)

    def _redraw_images(self):
        if not self._result:
            return
        if self._lock_exposure.get() and self._reference_y is None:
            self._reference_y = self._maximum_y(self._result)
        reference = self._reference_y if self._lock_exposure.get() else None
        self._display = display_images(self._result.xyz_fields, reference)
        self._photos.clear()
        for pixels,label in zip(self._display, self._image_labels):
            image = Image.fromarray(pixels); image.thumbnail((400, 280), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image); self._photos.append(photo); label.configure(image=photo, text="")
        if self._lab4_ready:
            self._draw_split()

    def _refresh_lab4(self):
        if self._result is None:
            self._clear_spectrum("Исправьте параметры сцены.")
            self._split_canvas.delete("all")
            self._comparison_info.set("Нет результата")
            return
        keys = tuple(self._result.spectra)
        self._entity.configure(values=tuple(entity_label(key) for key in keys))
        if self._selected.get() not in keys:
            self._selected.set(keys[0])
        self._entity_label.set(entity_label(self._selected.get()))
        self._draw_spectrum(); self._draw_split()

    def _draw_spectrum(self):
        if self._result is None: return
        key = self._selected.get(); data = self._result.spectra[key]; rec=data["reconstruction"]
        for v,value in zip(self._rgb_vars, rec.target_rgb): v.set(f"{value:.10g}")
        tolerance = max(float(np.max(np.abs(data['original']))), 1e-15) * 1e-12
        spectral_error = "≈ 0" if data['rmse'] <= tolerance else f"{data['rmse']:.3g}"
        color_error = "≈ 0" if rec.xyz_relative_error <= 1e-12 else f"{rec.xyz_relative_error:.3g}"
        self._spectrum_info.set(f"Ошибка спектра: {spectral_error} · Ошибка цвета: {color_error}")
        self._details = d = spectrum_details(key, data)
        xyz = lambda v: "(" + ", ".join(f"{x:.6g}" for x in v) + ")"
        above = f" · больше 1: {d['above_one_samples']}" if key == "surface" else ""
        self._calculation_info.set(
            f"XYZ цель = M·RGB: {xyz(d['target_xyz'])} · XYZ достигнутый: {xyz(d['actual_xyz'])} · "
            f"отн. ошибка: {d['xyz_relative_error']:.6g}\n"
            f"Отрицательных отсчётов: {d['negative_samples']}{above} · min: {d['min']:.6g} · max: {d['max']:.6g}")
        self._table.delete(*self._table.get_children())
        for nm,a,b,difference in zip(range(380,781,10), data["original"], rec.values, spectrum_display_delta(data)):
            self._table.insert("", "end", values=(nm, f"{a:.3g}",f"{b:.3g}",f"{difference:.3g}"))
        if self._spectrum_plot:
            self._spectrum_plot.get_tk_widget().destroy(); self._spectrum_plot.figure.clear()
        self._spectrum_plot = FigureCanvasTkAgg(spectrum_figure(key,data,d), master=self._spectrum_frame)
        self._spectrum_plot.get_tk_widget().pack(fill="both", expand=True); self._spectrum_plot.draw_idle()

    def _clear_spectrum(self, message):
        self._table.delete(*self._table.get_children())
        if self._spectrum_plot:
            self._spectrum_plot.get_tk_widget().destroy(); self._spectrum_plot.figure.clear(); self._spectrum_plot=None
        self._details = None
        self._calculation_info.set("")
        self._spectrum_info.set(message)

    def _draw_split(self):
        if not self._lab4_ready or self._result is None: return
        right = 1 if self._pair.get()=="image_1" else 2
        fraction = float(self._fraction.get())
        image = Image.fromarray(split_image(self._display[0], self._display[right], fraction))
        w,h = max(1,self._split_canvas.winfo_width()),max(1,self._split_canvas.winfo_height())
        image.thumbnail((w,h), Image.Resampling.LANCZOS)
        self._split_photo = ImageTk.PhotoImage(image)
        x,y=(w-image.width)//2,(h-image.height)//2
        self._split_bounds=(x,image.width)
        self._split_canvas.delete("all")
        self._split_canvas.create_image(x,y,image=self._split_photo,anchor="nw")
        border=x+round(image.width*fraction)
        self._split_canvas.create_line(border,y,border,y+image.height,fill="#ffd66e",width=2)
        self._split_canvas.create_text(x+8,y+8,text="Восстановленный",anchor="nw",fill="white")
        self._split_canvas.create_text(x+image.width-8,y+8,text=IMAGE_LABELS[self._pair.get()],anchor="ne",fill="white")
        metrics=self._result.metrics[f"A_vs_{right}"]
        self._comparison_info.set(f"Разделитель: {fraction:.0%} · Средняя ошибка: {metrics['display_mae']:.3g} · Максимальная: {metrics['display_max']:g}")

    def _drag(self,event):
        x,w=self._split_bounds
        self._fraction.set(min(1,max(0,(event.x-x)/max(w,1))))
        self._draw_split()

    def _apply_rgb(self):
        try:
            rgb=[float(v.get().replace(",",".")) for v in self._rgb_vars]
            from .reconstruction import reconstruct
            reconstruct(rgb, reflectance=self._selected.get()=="surface", method=self._method.get())
        except ValueError as error:
            message = str(error) if str(error).startswith("RGB") or "XYZ" in str(error) else "RGB: нужны три числа CIE RGB."
            self._clear_spectrum(f"Неверный RGB. {message}"); return
        self._rgb_overrides[self._selected.get()]=rgb
        self.recalculate(interactive=False)

    def _reset_rgb(self):
        self._rgb_overrides.pop(self._selected.get(), None)
        self.recalculate(interactive=False)

    def _spectrum_changed(self):
        self._rgb_overrides.clear()
        super()._spectrum_changed()

    def save_current(self):
        if not self.recalculate() or self._result is None: return
        directory=filedialog.askdirectory(title="Сохранить результаты")
        if directory:
            try:
                files=Lab4Service.save(self._result,Path(directory),self._fraction.get())
                self._status.set(f"Сохранено {len(files)} файлов: {directory}")
            except OSError as error: messagebox.showerror("Экспорт",str(error))

    def _load_parameters(self,p):
        self._loading=True
        try:
            for name,_ in SCENE_FIELDS: self._scene[name].set(f"{getattr(p,name):g}")
            self._reflectance=p.reflectance; self._full_visibility.set(p.require_full_visibility)
            for form in self._light_forms: form.destroy()
            self._light_forms.clear()
            for light in p.lights: self._add_light(light)
        finally: self._loading=False
        self._sync_view_azimuth_state(); self._reference_y=None
        self.recalculate(interactive=False)

    def load_lab3_white(self):
        self._rgb_overrides.clear()
        self._load_parameters(replace(SceneParameters.default().with_white_lights(),require_full_visibility=True))

    def load_scene(self):
        path=filedialog.askopenfilename(title="Открыть сцену",filetypes=[("Сцена","*.json")])
        if not path: return
        try:
            p,method,overrides=read_scene(path)
            from .reconstruction import METHODS
            if method not in METHODS: raise ValueError("Неизвестный метод")
            self._method.set(method); self._method_label.set(METHOD_LABELS[method]); self._rgb_overrides=overrides
            self._load_parameters(p)
        except (ValueError,KeyError,TypeError,OSError) as error: messagebox.showerror("Сцена",str(error))

    def reset(self):
        self._rgb_overrides.clear()
        super().reset()

    def _choose_method(self, _event=None):
        self._method.set(next(key for key,label in METHOD_LABELS.items() if label == self._method_label.get()))
        self.recalculate(interactive=False)

    def _choose_entity(self, _event=None):
        if self._result is None:
            return
        self._selected.set(next(key for key in self._result.spectra if entity_label(key) == self._entity_label.get()))
        self._draw_spectrum()

    def _choose_pair(self, _event=None):
        self._pair.set(next(key for key,label in IMAGE_LABELS.items() if label == self._pair_label.get()))
        self._draw_split()

    def _show_summary(self):
        stats = self._result.statistics
        lines = ["Яркость, Вт/(м²·ср)"]
        for index, point in enumerate(stats.probes, 1):
            lines.append(f"Точка {index}: {point.luminance_w_m2_sr:.6g}")
        lines += [f"Минимум: {stats.minimum.luminance_w_m2_sr:.6g}",
                  f"Максимум: {stats.maximum.luminance_w_m2_sr:.6g}"]
        if any(data['reconstruction'].negative_samples for data in self._result.spectra.values()):
            lines.append("Спектр содержит отрицательные значения.")
        if np.any(self._result.spectra['surface']['reconstruction'].values > 1+1e-9):
            lines.append("Коэффициент отражения превышает 1.")
        self._report.configure(state="normal")
        self._report.delete("1.0", "end")
        self._report.insert("1.0", "\n".join(lines))
        self._report.configure(state="disabled")

    def _update_light_directions(self, parameters):
        for form, light in zip(self._light_forms, parameters.lights):
            form.direction_status.set("" if light.enabled else "Выключен")

    def destroy(self):
        if self._job:
            self.after_cancel(self._job); self._job=None
        super().destroy()
