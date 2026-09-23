"""Графический интерфейс пользователя на Tkinter.

Модуль отвечает только за ввод параметров, отображение результата и вызов
сценария расчёта. Физические формулы находятся в модулях geometry и physics.

Окно состоит из:
    * левой панели: параметры сцены, свойства материала, список источников
      света (добавление, удаление, включение/выключение) и кнопки управления;
    * области визуализации: изображение распределения яркости и сечения;
    * панели числовых результатов, требуемых в отчёте, и строки состояния.

При включённом автопересчёте изображение обновляется вскоре после каждого
изменения параметров; ошибки ввода в этом режиме выводятся в строку
состояния, а не всплывающим окном, чтобы не мешать набору значения.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .models import (
    MAXIMUM_LIGHT_COUNT,
    BlinnPhongMaterial,
    LightSource,
    ParameterValidationError,
    SceneParameters,
)
from .service import ComputationResult, LuminanceService

DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent.parent / "output"
DEFAULT_IMAGE_NAME = "sphere_luminance.png"

# Задержка автопересчёта после последнего изменения поля, мс: пользователь
# успевает набрать число целиком, а отклик остаётся мгновенным на глаз.
AUTO_RECALCULATE_DELAY_MS = 350

# Палитры для просмотра на экране. В файл всегда пишется полутоновое
# изображение 0-255, как требует задание.
DISPLAY_COLORMAPS = ("gray", "inferno", "viridis", "jet")

# Параметры источника, добавляемого кнопкой «+ источник».
NEW_LIGHT = LightSource(x_mm=0.0, y_mm=0.0, z_mm=2000.0, intensity_w_sr=500.0)

HINT_COLOUR = "#666666"


@dataclass(frozen=True)
class FieldSpecification:
    """Описание одного числового поля ввода.

    Attributes:
        attribute: Имя соответствующего поля dataclass-модели.
        label: Подпись поля в интерфейсе.
        hint: Подсказка о допустимых значениях.
        is_integer: True, если значение должно быть целым.
    """

    attribute: str
    label: str
    hint: str = ""
    is_integer: bool = False


SCENE_FIELDS: tuple[FieldSpecification, ...] = (
    FieldSpecification("height_mm", "Высота экрана H", "100…10000 мм"),
    FieldSpecification("width_mm", "Ширина экрана W", "100…10000 мм"),
    FieldSpecification("height_px", "Разрешение Hres", "200…800 пикс", is_integer=True),
    FieldSpecification("width_px", "Разрешение Wres", "200…800 пикс", is_integer=True),
    FieldSpecification("observer_z_mm", "Наблюдатель zO", "> zC + R, мм"),
    FieldSpecification("sphere_x_mm", "Центр сферы xC", "±10000 мм"),
    FieldSpecification("sphere_y_mm", "Центр сферы yC", "±10000 мм"),
    FieldSpecification("sphere_z_mm", "Центр сферы zC", "100…10000 мм"),
    FieldSpecification("sphere_radius_mm", "Радиус сферы R", "> 0 мм"),
)

MATERIAL_FIELDS: tuple[FieldSpecification, ...] = (
    FieldSpecification("diffuse", "Диффузное kd", "0…10"),
    FieldSpecification("specular", "Зеркальное ks", "0…10"),
    FieldSpecification("shininess", "Блеск ke", "1…10000"),
)

LIGHT_FIELDS: tuple[FieldSpecification, ...] = (
    FieldSpecification("x_mm", "xL, мм"),
    FieldSpecification("y_mm", "yL, мм"),
    FieldSpecification("z_mm", "zL, мм"),
    FieldSpecification("intensity_w_sr", "I0, Вт/ср"),
)


def parse_number(raw_text: str, specification: FieldSpecification, owner: str = "") -> float | int:
    """Преобразует текст поля ввода в число; допускается десятичная запятая.

    Args:
        raw_text: Содержимое поля ввода.
        specification: Описание поля, задающее требуемый тип значения.
        owner: Префикс для сообщения об ошибке (например, «Источник 2»).

    Returns:
        Числовое значение параметра.

    Raises:
        ParameterValidationError: Если текст не является числом нужного типа.
    """
    title = f"{owner}: {specification.label}" if owner else specification.label
    text = raw_text.strip().replace(",", ".")
    if not text:
        raise ParameterValidationError(f"{title}: значение не задано.")

    try:
        number = float(text)
    except ValueError as error:
        raise ParameterValidationError(f"{title}: «{raw_text}» не является числом.") from error

    if specification.is_integer:
        if not number.is_integer():
            raise ParameterValidationError(f"{title}: требуется целое число.")
        return int(number)
    return number


class NumberForm(ttk.LabelFrame):
    """Группа подписанных числовых полей, связанная с dataclass-моделью."""

    def __init__(
        self,
        master: tk.Misc,
        title: str,
        specifications: tuple[FieldSpecification, ...],
        on_change: Callable[[], None],
    ) -> None:
        """Создаёт поля ввода.

        Args:
            master: Родительский контейнер.
            title: Заголовок группы.
            specifications: Описания полей.
            on_change: Вызывается при любом изменении значения поля.
        """
        super().__init__(master, text=title, padding=6)
        self._specifications = specifications
        self._variables: dict[str, tk.StringVar] = {}

        for row, specification in enumerate(specifications):
            variable = tk.StringVar()
            variable.trace_add("write", lambda *_: on_change())
            self._variables[specification.attribute] = variable

            ttk.Label(self, text=specification.label).grid(row=row, column=0, sticky="w", pady=1)
            ttk.Entry(self, textvariable=variable, width=10, justify="right").grid(
                row=row, column=1, sticky="ew", padx=4, pady=1
            )
            ttk.Label(self, text=specification.hint, foreground=HINT_COLOUR).grid(
                row=row, column=2, sticky="w", pady=1
            )

    def fill(self, source: Any) -> None:
        """Записывает в поля значения одноимённых атрибутов объекта."""
        for specification in self._specifications:
            value = getattr(source, specification.attribute)
            self._variables[specification.attribute].set(f"{value:g}")

    def read(self) -> dict[str, float | int]:
        """Считывает значения полей.

        Raises:
            ParameterValidationError: Если поле пусто или содержит не число.
        """
        return {
            specification.attribute: parse_number(
                self._variables[specification.attribute].get(), specification
            )
            for specification in self._specifications
        }


class LightRow:
    """Строка таблицы источников: флажок «вкл», четыре поля и кнопка удаления."""

    def __init__(
        self,
        master: ttk.Frame,
        light: LightSource,
        on_change: Callable[[], None],
        on_remove: Callable[["LightRow"], None],
    ) -> None:
        """Создаёт виджеты строки (размещаются методом place_at).

        Args:
            master: Контейнер таблицы источников.
            light: Начальные параметры источника.
            on_change: Вызывается при изменении любого значения строки.
            on_remove: Вызывается при нажатии кнопки удаления.
        """
        self.enabled = tk.BooleanVar(value=light.enabled)
        self.enabled.trace_add("write", lambda *_: on_change())
        self._variables: dict[str, tk.StringVar] = {}

        self._number_label = ttk.Label(master)
        self._widgets: list[tk.Widget] = [
            self._number_label,
            ttk.Checkbutton(master, variable=self.enabled),
        ]
        for specification in LIGHT_FIELDS:
            variable = tk.StringVar(value=f"{getattr(light, specification.attribute):g}")
            variable.trace_add("write", lambda *_: on_change())
            self._variables[specification.attribute] = variable
            self._widgets.append(ttk.Entry(master, textvariable=variable, width=7, justify="right"))
        self._widgets.append(ttk.Button(master, text="✕", width=2, command=lambda: on_remove(self)))

    def place_at(self, row: int) -> None:
        """Размещает строку в таблице и обновляет её номер."""
        self._number_label.configure(text=f"{row}")
        for column, widget in enumerate(self._widgets):
            widget.grid(row=row, column=column, padx=1, pady=1)

    def destroy(self) -> None:
        """Удаляет виджеты строки."""
        for widget in self._widgets:
            widget.destroy()

    def read(self, number: int) -> LightSource:
        """Считывает параметры источника.

        Args:
            number: Порядковый номер источника для сообщений об ошибке.

        Raises:
            ParameterValidationError: Если поле пусто или содержит не число.
        """
        values = {
            specification.attribute: parse_number(
                self._variables[specification.attribute].get(), specification, f"Источник {number}"
            )
            for specification in LIGHT_FIELDS
        }
        return LightSource(**values, enabled=self.enabled.get())  # type: ignore[arg-type]


class LightsPanel(ttk.LabelFrame):
    """Редактируемый список точечных источников света."""

    def __init__(self, master: tk.Misc, on_change: Callable[[], None]) -> None:
        """Создаёт заголовок таблицы и кнопку добавления источника.

        Args:
            master: Родительский контейнер.
            on_change: Вызывается при любом изменении списка источников.
        """
        super().__init__(master, text="Источники света (I = I0·cos θ)", padding=6)
        self._on_change = on_change
        self._rows: list[LightRow] = []

        self._table = ttk.Frame(self)
        self._table.pack(fill="x")
        headers = ("№", "вкл", *(spec.label for spec in LIGHT_FIELDS), "")
        for column, header in enumerate(headers):
            ttk.Label(self._table, text=header, foreground=HINT_COLOUR).grid(row=0, column=column)

        self._add_button = ttk.Button(self, text="+ источник", command=self._add_default)
        self._add_button.pack(anchor="w", pady=(4, 0))

    def fill(self, lights: tuple[LightSource, ...]) -> None:
        """Заменяет таблицу заданным набором источников."""
        for row in self._rows:
            row.destroy()
        self._rows = [LightRow(self._table, light, self._on_change, self._remove) for light in lights]
        self._relayout()

    def read(self) -> tuple[LightSource, ...]:
        """Считывает все источники таблицы."""
        return tuple(row.read(number) for number, row in enumerate(self._rows, start=1))

    def _add_default(self) -> None:
        """Добавляет источник с параметрами по умолчанию."""
        self._rows.append(LightRow(self._table, NEW_LIGHT, self._on_change, self._remove))
        self._relayout()
        self._on_change()

    def _remove(self, row: LightRow) -> None:
        """Удаляет строку источника."""
        row.destroy()
        self._rows.remove(row)
        self._relayout()
        self._on_change()

    def _relayout(self) -> None:
        """Перенумеровывает строки и блокирует добавление сверх максимума."""
        for number, row in enumerate(self._rows, start=1):
            row.place_at(number)
        state = "normal" if len(self._rows) < MAXIMUM_LIGHT_COUNT else "disabled"
        self._add_button.configure(state=state)


class ResultCanvas(ttk.Frame):
    """Область визуализации: изображение распределения яркости и сечения."""

    def __init__(self, master: tk.Misc) -> None:
        """Создаёт полотно matplotlib с двумя графиками."""
        super().__init__(master)
        self._figure = Figure(figsize=(10, 4.8), dpi=100)
        self._image_axes = self._figure.add_subplot(1, 2, 1)
        self._section_axes = self._figure.add_subplot(1, 2, 2)
        self._colorbar = None

        self._canvas = FigureCanvasTkAgg(self._figure, master=self)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

    def show(self, result: ComputationResult, colormap: str) -> None:
        """Перерисовывает изображение и сечения.

        Args:
            result: Результат расчёта.
            colormap: Палитра для отображения нормированного изображения.
        """
        self._draw_image(result, colormap)
        self._draw_sections(result)
        self._figure.tight_layout()
        self._canvas.draw_idle()

    def _draw_image(self, result: ComputationResult, colormap: str) -> None:
        """Выводит нормированное изображение (ось Y — по горизонтали, X — по вертикали)."""
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None
        self._image_axes.clear()

        parameters = result.parameters
        y_minimum_mm, y_maximum_mm = parameters.y_bounds_mm
        x_minimum_mm, x_maximum_mm = parameters.x_bounds_mm
        image = self._image_axes.imshow(
            result.grey_levels,
            cmap=colormap,
            vmin=0,
            vmax=255,
            origin="upper",
            extent=(y_minimum_mm, y_maximum_mm, x_minimum_mm, x_maximum_mm),
            aspect="equal",
        )

        maximum_x, maximum_y = result.statistics.maximum.screen_mm
        self._image_axes.plot(maximum_y, maximum_x, "+", color="tab:red", markersize=12)

        self._image_axes.set_title("Яркость на сфере (0–255), + — максимум")
        self._image_axes.set_xlabel("y, мм")
        self._image_axes.set_ylabel("x, мм")
        self._colorbar = self._figure.colorbar(image, ax=self._image_axes, fraction=0.046, pad=0.04)
        self._colorbar.set_label("Градации серого")

    def _draw_sections(self, result: ComputationResult) -> None:
        """Выводит сечения через проекцию центра сферы."""
        self._section_axes.clear()
        for section in result.cross_sections:
            self._section_axes.plot(section.coordinates_mm, section.values_w_m2_sr, label=section.title)

        self._section_axes.set_title("Сечения через центр сферы")
        self._section_axes.set_xlabel("Координата на экране, мм")
        self._section_axes.set_ylabel("Яркость L, Вт/(м²·ср)")
        self._section_axes.grid(True, linestyle=":", linewidth=0.7)
        self._section_axes.legend()


class ApplicationWindow(tk.Tk):
    """Главное окно приложения расчёта яркости на сфере."""

    def __init__(self) -> None:
        """Собирает интерфейс и выполняет расчёт с параметрами по умолчанию."""
        super().__init__()
        self.title("ЛР №2. Расчёт яркости на сфере от точечных источников света")
        self.geometry("1440x880")
        self.minsize(1150, 760)

        self._result: ComputationResult | None = None
        self._pending_job: str | None = None
        self._loading = False
        self._auto_recalculate = tk.BooleanVar(value=True)
        self._colormap = tk.StringVar(value=DISPLAY_COLORMAPS[0])
        self._status = tk.StringVar()

        content = ttk.Frame(self, padding=8)
        content.pack(fill="both", expand=True)

        controls = ttk.Frame(content)
        controls.pack(side="left", fill="y")
        self._scene_form = NumberForm(controls, "Экран, наблюдатель, сфера", SCENE_FIELDS, self._schedule)
        self._scene_form.pack(fill="x")
        self._material_form = NumberForm(
            controls, "Материал: f = kd + ks·(h·N)^ke", MATERIAL_FIELDS, self._schedule
        )
        self._material_form.pack(fill="x", pady=(6, 0))
        self._lights_panel = LightsPanel(controls, self._schedule)
        self._lights_panel.pack(fill="x", pady=(6, 0))
        self._build_buttons(controls)

        right_panel = ttk.Frame(content)
        right_panel.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self._canvas = ResultCanvas(right_panel)
        self._canvas.pack(fill="both", expand=True)

        results = ttk.LabelFrame(right_panel, text="Результаты расчёта", padding=6)
        results.pack(fill="x", pady=(6, 0))
        self._report = tk.Text(results, height=8, wrap="none", font=("Consolas", 9), state="disabled")
        self._report.pack(fill="x")

        ttk.Label(self, textvariable=self._status, anchor="w", padding=(8, 2)).pack(fill="x")

        self.bind("<Return>", lambda _: self.recalculate())
        self.reset_parameters()

    def _build_buttons(self, master: ttk.Frame) -> None:
        """Размещает кнопки и настройки отображения под панелями параметров."""
        frame = ttk.Frame(master)
        frame.pack(fill="x", pady=(8, 0))

        ttk.Checkbutton(frame, text="Автопересчёт", variable=self._auto_recalculate).pack(anchor="w")
        palette = ttk.Frame(frame)
        palette.pack(fill="x", pady=4)
        ttk.Label(palette, text="Палитра просмотра:").pack(side="left")
        combobox = ttk.Combobox(
            palette, textvariable=self._colormap, values=DISPLAY_COLORMAPS, state="readonly", width=10
        )
        combobox.pack(side="left", padx=4)
        combobox.bind("<<ComboboxSelected>>", lambda _: self._redraw())

        ttk.Button(frame, text="Рассчитать (Enter)", command=self.recalculate).pack(fill="x", pady=2)
        ttk.Button(frame, text="Сохранить изображение и значения", command=self.save_result).pack(
            fill="x", pady=2
        )
        ttk.Button(frame, text="Значения по умолчанию", command=self.reset_parameters).pack(
            fill="x", pady=2
        )

    def read_parameters(self) -> SceneParameters:
        """Собирает параметры сцены из всех панелей.

        Raises:
            ParameterValidationError: Если какое-либо поле некорректно.
        """
        material = BlinnPhongMaterial(**self._material_form.read())
        return SceneParameters(
            **self._scene_form.read(),  # type: ignore[arg-type]
            material=material,
            lights=self._lights_panel.read(),
        )

    def recalculate(self, *, interactive: bool = True) -> None:
        """Считывает параметры, выполняет расчёт и обновляет отображение.

        Args:
            interactive: True — ошибки показываются окном сообщения, False
                (автопересчёт) — только в строке состояния.
        """
        if self._pending_job is not None:
            self.after_cancel(self._pending_job)
            self._pending_job = None
        try:
            result = LuminanceService.compute(self.read_parameters())
        except ValueError as error:
            self._status.set(f"⚠ {error}")
            if interactive:
                messagebox.showerror("Некорректные параметры", str(error))
            return

        self._result = result
        self._redraw()
        self._set_report(result.report)
        maximum = result.statistics.maximum.luminance_w_m2_sr
        self._status.set(f"Расчёт выполнен. Lmax = {maximum:.4e} Вт/(м²·ср) → 255.")

    def save_result(self) -> None:
        """Сохраняет изображение (PNG) и файл расчётных значений рядом с ним."""
        if self._result is None:
            messagebox.showinfo("Нет данных", "Сначала выполните расчёт.")
            return

        DEFAULT_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        selected_path = filedialog.asksaveasfilename(
            title="Сохранить изображение распределения яркости",
            initialdir=str(DEFAULT_OUTPUT_DIRECTORY),
            initialfile=DEFAULT_IMAGE_NAME,
            defaultextension=".png",
            filetypes=[("Изображение PNG", "*.png"), ("Все файлы", "*.*")],
        )
        if not selected_path:
            return

        image_path, report_path = LuminanceService.save(self._result, Path(selected_path))
        messagebox.showinfo("Сохранено", f"Изображение:\n{image_path}\n\nЗначения:\n{report_path}")

    def reset_parameters(self) -> None:
        """Возвращает все поля к значениям по умолчанию и пересчитывает."""
        defaults = SceneParameters.default()
        self._loading = True
        self._scene_form.fill(defaults)
        self._material_form.fill(defaults.material)
        self._lights_panel.fill(defaults.lights)
        self._loading = False
        self.recalculate()

    def _schedule(self) -> None:
        """Откладывает автопересчёт до паузы во вводе."""
        if self._loading or not self._auto_recalculate.get():
            return
        if self._pending_job is not None:
            self.after_cancel(self._pending_job)
        self._pending_job = self.after(
            AUTO_RECALCULATE_DELAY_MS, lambda: self.recalculate(interactive=False)
        )

    def _redraw(self) -> None:
        """Перерисовывает последний результат в выбранной палитре."""
        if self._result is not None:
            self._canvas.show(self._result, self._colormap.get())

    def _set_report(self, text: str) -> None:
        """Выводит текст отчёта в панель результатов."""
        self._report.configure(state="normal")
        self._report.delete("1.0", tk.END)
        self._report.insert("1.0", text)
        self._report.configure(state="disabled")
