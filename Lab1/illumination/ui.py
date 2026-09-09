"""Графический интерфейс пользователя на Tkinter.

Модуль отвечает только за ввод параметров, отображение результата на экране
и вызов сценария расчёта. Физические формулы и обработка данных находятся в
модулях physics, analysis и rendering.

Окно состоит из трёх частей:
    * панель параметров с проверкой введённых значений;
    * область визуализации: нормированное изображение распределения
      освещённости и график сечений через центр круга;
    * панель числовых результатов, требуемых в отчёте.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .models import ParameterValidationError, SceneParameters
from .service import ComputationResult, IlluminanceService

# Каталог, предлагаемый по умолчанию при сохранении изображения.
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent.parent / "output"
DEFAULT_IMAGE_NAME = "illuminance.png"


@dataclass(frozen=True)
class FieldSpecification:
    """Описание одного поля ввода панели параметров.

    Attributes:
        attribute: Имя соответствующего поля SceneParameters.
        label: Подпись поля в интерфейсе.
        hint: Подсказка о допустимых значениях.
        is_integer: True, если значение должно быть целым (разрешение).
    """

    attribute: str
    label: str
    hint: str
    is_integer: bool = False


# Порядок полей соответствует порядку перечисления исходных данных в задании.
FIELD_SPECIFICATIONS: tuple[FieldSpecification, ...] = (
    FieldSpecification("height_mm", "Высота области H", "100…10000 мм"),
    FieldSpecification("width_mm", "Ширина области W", "100…10000 мм"),
    FieldSpecification("height_px", "Разрешение Hres", "200…800 пикс", is_integer=True),
    FieldSpecification("width_px", "Разрешение Wres", "200…800 пикс", is_integer=True),
    FieldSpecification("light_x_mm", "Источник xL", "±10000 мм"),
    FieldSpecification("light_y_mm", "Источник yL", "±10000 мм"),
    FieldSpecification("light_z_mm", "Источник zL", "100…10000 мм"),
    FieldSpecification("axial_intensity_w_sr", "Сила излучения I0", "0.01…10000 Вт/ср"),
    FieldSpecification("circle_center_x_mm", "Центр круга xc", "мм"),
    FieldSpecification("circle_center_y_mm", "Центр круга yc", "мм"),
    FieldSpecification("circle_radius_mm", "Радиус круга R", "> 0 мм"),
)


class ParameterForm(ttk.LabelFrame):
    """Панель ввода исходных данных сцены."""

    def __init__(self, master: tk.Misc) -> None:
        """Создаёт поля ввода и заполняет их значениями по умолчанию.

        Args:
            master: Родительский контейнер Tkinter.
        """
        super().__init__(master, text="Параметры сцены", padding=8)
        self._variables: dict[str, tk.StringVar] = {}
        self._build_rows()
        self.fill(SceneParameters.default())

    def _build_rows(self) -> None:
        """Размещает подписи, поля ввода и подсказки о пределах значений."""
        for row, specification in enumerate(FIELD_SPECIFICATIONS):
            variable = tk.StringVar()
            self._variables[specification.attribute] = variable

            ttk.Label(self, text=specification.label).grid(
                row=row, column=0, sticky="w", padx=(0, 6), pady=2
            )
            ttk.Entry(self, textvariable=variable, width=12, justify="right").grid(
                row=row, column=1, sticky="ew", pady=2
            )
            ttk.Label(self, text=specification.hint, foreground="#666666").grid(
                row=row, column=2, sticky="w", padx=(6, 0), pady=2
            )

    def fill(self, parameters: SceneParameters) -> None:
        """Записывает значения параметров в поля ввода.

        Args:
            parameters: Набор параметров, отображаемый в форме.
        """
        for specification in FIELD_SPECIFICATIONS:
            value = getattr(parameters, specification.attribute)
            self._variables[specification.attribute].set(f"{value:g}")

    def read(self) -> SceneParameters:
        """Считывает параметры из полей ввода.

        Returns:
            Набор параметров, введённый пользователем.

        Raises:
            ParameterValidationError: Если поле пусто или содержит не число.
        """
        values: dict[str, float | int] = {}
        for specification in FIELD_SPECIFICATIONS:
            raw_text = self._variables[specification.attribute].get()
            values[specification.attribute] = self._parse(raw_text, specification)

        return SceneParameters(**values)  # type: ignore[arg-type]

    @staticmethod
    def _parse(raw_text: str, specification: FieldSpecification) -> float | int:
        """Преобразует текст поля ввода в число.

        Десятичная запятая допускается наравне с точкой: это привычный ввод
        для русской раскладки клавиатуры.

        Args:
            raw_text: Содержимое поля ввода.
            specification: Описание поля, задающее требуемый тип значения.

        Returns:
            Числовое значение параметра.

        Raises:
            ParameterValidationError: Если текст не является числом нужного типа.
        """
        text = raw_text.strip().replace(",", ".")
        if not text:
            raise ParameterValidationError(
                f"{specification.label}: значение не задано."
            )

        try:
            number = float(text)
        except ValueError as error:
            raise ParameterValidationError(
                f"{specification.label}: «{raw_text}» не является числом."
            ) from error

        if specification.is_integer:
            if not number.is_integer():
                raise ParameterValidationError(
                    f"{specification.label}: разрешение задаётся целым числом "
                    "пикселей."
                )
            return int(number)

        return number


class ResultCanvas(ttk.Frame):
    """Область визуализации: изображение распределения и график сечений."""

    def __init__(self, master: tk.Misc) -> None:
        """Создаёт полотно matplotlib с двумя графиками.

        Args:
            master: Родительский контейнер Tkinter.
        """
        super().__init__(master)
        self._figure = Figure(figsize=(9.5, 4.6), dpi=100)
        self._image_axes = self._figure.add_subplot(1, 2, 1)
        self._section_axes = self._figure.add_subplot(1, 2, 2)
        self._colorbar = None

        self._canvas = FigureCanvasTkAgg(self._figure, master=self)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

    def show(self, result: ComputationResult) -> None:
        """Перерисовывает изображение распределения и график сечений.

        Args:
            result: Результат расчёта, подлежащий отображению.
        """
        self._draw_image(result)
        self._draw_sections(result)
        self._figure.tight_layout()
        self._canvas.draw()

    def _draw_image(self, result: ComputationResult) -> None:
        """Выводит нормированное изображение распределения освещённости.

        По горизонтали изображения отложена ось Y (ширина W), по вертикали —
        ось X (высота H), что соответствует принятой системе координат.

        Args:
            result: Результат расчёта.
        """
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None

        self._image_axes.clear()

        parameters = result.parameters
        y_minimum_mm, y_maximum_mm = parameters.y_bounds_mm
        x_minimum_mm, x_maximum_mm = parameters.x_bounds_mm

        image = self._image_axes.imshow(
            result.grey_levels,
            cmap="gray",
            vmin=0,
            vmax=255,
            origin="upper",
            extent=(y_minimum_mm, y_maximum_mm, x_minimum_mm, x_maximum_mm),
            aspect="equal",
        )

        self._image_axes.set_title("Распределение освещённости (0–255)")
        self._image_axes.set_xlabel("y, мм")
        self._image_axes.set_ylabel("x, мм")
        self._colorbar = self._figure.colorbar(
            image, ax=self._image_axes, fraction=0.046, pad=0.04
        )
        self._colorbar.set_label("Градации серого")

    def _draw_sections(self, result: ComputationResult) -> None:
        """Выводит график сечений, проходящих через центр расчётной области.

        Args:
            result: Результат расчёта.
        """
        self._section_axes.clear()

        for section in result.cross_sections:
            self._section_axes.plot(
                section.coordinates_mm, section.values_w_m2, label=section.title
            )

        self._section_axes.set_title("Сечения через центр круга")
        self._section_axes.set_xlabel("Координата, мм")
        self._section_axes.set_ylabel("Освещённость E, Вт/м²")
        self._section_axes.grid(True, linestyle=":", linewidth=0.7)
        self._section_axes.legend()


class StatisticsPanel(ttk.LabelFrame):
    """Панель числовых результатов, требуемых в отчёте по работе."""

    def __init__(self, master: tk.Misc) -> None:
        """Создаёт текстовое поле для вывода результатов расчёта.

        Args:
            master: Родительский контейнер Tkinter.
        """
        super().__init__(master, text="Результаты расчёта", padding=8)
        self._text = tk.Text(self, height=9, wrap="none", font=("Consolas", 9))
        self._text.pack(fill="both", expand=True)
        self._text.configure(state="disabled")

    def show(self, result: ComputationResult) -> None:
        """Заполняет панель значениями освещённости и статистикой.

        Args:
            result: Результат расчёта.
        """
        self._text.configure(state="normal")
        self._text.delete("1.0", tk.END)
        self._text.insert("1.0", self._format(result))
        self._text.configure(state="disabled")

    @staticmethod
    def _format(result: ComputationResult) -> str:
        """Формирует текстовое представление результатов расчёта.

        Args:
            result: Результат расчёта.

        Returns:
            Многострочный текст с контрольными точками и статистикой.
        """
        statistics = result.statistics
        lines = ["Освещённость в контрольных точках, Вт/м²:"]
        lines.extend(
            f"  {probe.title:<16} x = {probe.x_mm:>9.3f} мм, "
            f"y = {probe.y_mm:>9.3f} мм:  {probe.illuminance_w_m2:.6e}"
            for probe in statistics.probes
        )
        lines.append("")
        lines.append(
            "Внутри круга (пикселей: "
            f"{statistics.pixel_count}):  "
            f"максимум {statistics.maximum_w_m2:.6e}  "
            f"минимум {statistics.minimum_w_m2:.6e}  "
            f"среднее {statistics.mean_w_m2:.6e}  Вт/м²"
        )

        return "\n".join(lines)


class ApplicationWindow(tk.Tk):
    """Главное окно приложения расчёта освещённости."""

    def __init__(self) -> None:
        """Собирает интерфейс и выполняет расчёт с параметрами по умолчанию."""
        super().__init__()
        self.title("ЛР №1. Расчёт освещённости на плоскости от точечного источника")
        self.geometry("1240x760")
        self.minsize(1000, 640)

        self._result: ComputationResult | None = None

        content = ttk.Frame(self, padding=10)
        content.pack(fill="both", expand=True)

        self._form = ParameterForm(content)
        self._form.pack(side="left", fill="y")

        right_panel = ttk.Frame(content)
        right_panel.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self._canvas = ResultCanvas(right_panel)
        self._canvas.pack(fill="both", expand=True)

        self._statistics = StatisticsPanel(right_panel)
        self._statistics.pack(fill="x", pady=(8, 0))

        self._build_buttons()
        self.recalculate()

    def _build_buttons(self) -> None:
        """Размещает кнопки управления под панелью параметров."""
        buttons = ttk.Frame(self._form)
        buttons.grid(
            row=len(FIELD_SPECIFICATIONS), column=0, columnspan=3, sticky="ew", pady=(12, 0)
        )

        ttk.Button(buttons, text="Рассчитать", command=self.recalculate).pack(
            fill="x", pady=2
        )
        ttk.Button(
            buttons, text="Сохранить изображение", command=self.save_image
        ).pack(fill="x", pady=2)
        ttk.Button(
            buttons, text="Значения по умолчанию", command=self.reset_parameters
        ).pack(fill="x", pady=2)

    def recalculate(self) -> None:
        """Считывает параметры, выполняет расчёт и обновляет отображение.

        Ошибки ввода и недопустимые сочетания параметров показываются
        пользователю как сообщение, не прерывая работу приложения.
        """
        try:
            parameters = self._form.read()
            result = IlluminanceService.compute(parameters)
        except (ParameterValidationError, ValueError) as error:
            messagebox.showerror("Некорректные параметры", str(error))
            return

        self._result = result
        self._canvas.show(result)
        self._statistics.show(result)

    def save_image(self) -> None:
        """Сохраняет нормированное изображение распределения в файл."""
        if self._result is None:
            messagebox.showinfo("Нет данных", "Сначала выполните расчёт.")
            return

        DEFAULT_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        selected_path = filedialog.asksaveasfilename(
            title="Сохранить изображение распределения",
            initialdir=str(DEFAULT_OUTPUT_DIRECTORY),
            initialfile=DEFAULT_IMAGE_NAME,
            defaultextension=".png",
            filetypes=[("Изображение PNG", "*.png"), ("Все файлы", "*.*")],
        )
        if not selected_path:
            return

        saved_path = IlluminanceService.save_image(self._result, Path(selected_path))
        messagebox.showinfo("Изображение сохранено", f"Файл записан:\n{saved_path}")

    def reset_parameters(self) -> None:
        """Возвращает поля ввода к значениям по умолчанию и пересчитывает."""
        self._form.fill(SceneParameters.default())
        self.recalculate()
