"""Описание параметров сцены и проверка их допустимости.

Модуль отвечает только за структуру входных данных задачи и за контроль
рекомендованных пределов. Расчётов освещённости здесь нет.

Принятая система координат (соответствует рисунку из постановки задачи):

    * приёмная плоскость совпадает с плоскостью z = 0;
    * начало координат O расположено в центре прямоугольной области
      изображения размером H x W [мм];
    * ось X направлена вдоль высоты области H и отображается по вертикали
      изображения (Hres пикселей);
    * ось Y направлена вдоль ширины области W и отображается по горизонтали
      изображения (Wres пикселей);
    * ось Z направлена вверх, в сторону источника света L(xL, yL, zL).
"""

from __future__ import annotations

from dataclasses import dataclass

# Относительный допуск при проверке квадратности пикселя: расчётная сетка
# считается квадратной, если размеры пикселя по осям различаются менее чем
# на 0.1 % (пользователь вводит целые пиксели и миллиметры, поэтому точное
# равенство недостижимо).
SQUARE_PIXEL_RELATIVE_TOLERANCE = 1e-3


class ParameterValidationError(ValueError):
    """Ошибка недопустимого значения параметра сцены.

    Отдельный тип исключения позволяет интерфейсу пользователя отличать
    ошибки ввода от программных сбоев и показывать их пользователю как
    сообщение, а не как аварийное завершение.
    """


@dataclass(frozen=True)
class ValueRange:
    """Замкнутый диапазон допустимых значений одного параметра.

    Attributes:
        minimum: Нижняя граница диапазона (включительно).
        maximum: Верхняя граница диапазона (включительно).
        unit: Единица измерения параметра для сообщений об ошибке.
    """

    minimum: float
    maximum: float
    unit: str

    def contains(self, value: float) -> bool:
        """Проверяет попадание значения в диапазон.

        Args:
            value: Проверяемое значение параметра.

        Returns:
            True, если minimum <= value <= maximum.
        """
        return self.minimum <= value <= self.maximum

    def describe(self) -> str:
        """Возвращает текстовое описание диапазона для сообщений об ошибке."""
        return f"от {self.minimum:g} до {self.maximum:g} {self.unit}"


class ParameterLimits:
    """Рекомендованные пределы значений параметров из постановки задачи."""

    AREA_SIZE = ValueRange(100.0, 10000.0, "мм")
    RESOLUTION = ValueRange(200, 800, "пикс")
    LIGHT_XY = ValueRange(-10000.0, 10000.0, "мм")
    LIGHT_Z = ValueRange(100.0, 10000.0, "мм")
    RADIANT_INTENSITY = ValueRange(0.01, 10000.0, "Вт/ср")


@dataclass(frozen=True)
class SceneParameters:
    """Полный набор исходных данных для расчёта распределения освещённости.

    Attributes:
        height_mm: Высота области изображения H вдоль оси X, мм.
        width_mm: Ширина области изображения W вдоль оси Y, мм.
        height_px: Разрешение изображения по высоте Hres, пикселей.
        width_px: Разрешение изображения по ширине Wres, пикселей.
        light_x_mm: Координата xL источника света, мм.
        light_y_mm: Координата yL источника света, мм.
        light_z_mm: Высота zL источника света над плоскостью, мм.
        axial_intensity_w_sr: Сила излучения I0 в направлении оси диаграммы
            (theta = 0), Вт/ср.
        circle_center_x_mm: Координата x центра расчётного круга, мм.
        circle_center_y_mm: Координата y центра расчётного круга, мм.
        circle_radius_mm: Радиус расчётного круга R, мм.
    """

    height_mm: float
    width_mm: float
    height_px: int
    width_px: int
    light_x_mm: float
    light_y_mm: float
    light_z_mm: float
    axial_intensity_w_sr: float
    circle_center_x_mm: float
    circle_center_y_mm: float
    circle_radius_mm: float

    @property
    def pixel_height_mm(self) -> float:
        """Линейный размер пикселя вдоль оси X: H / Hres, мм."""
        return self.height_mm / self.height_px

    @property
    def pixel_width_mm(self) -> float:
        """Линейный размер пикселя вдоль оси Y: W / Wres, мм."""
        return self.width_mm / self.width_px

    @property
    def x_bounds_mm(self) -> tuple[float, float]:
        """Границы области по оси X: (-H/2, +H/2), мм."""
        return -self.height_mm / 2.0, self.height_mm / 2.0

    @property
    def y_bounds_mm(self) -> tuple[float, float]:
        """Границы области по оси Y: (-W/2, +W/2), мм."""
        return -self.width_mm / 2.0, self.width_mm / 2.0

    @staticmethod
    def default() -> "SceneParameters":
        """Возвращает корректный набор параметров, используемый при запуске.

        Значения выбраны внутри рекомендованных пределов и дают квадратный
        пиксель (1000 мм / 500 пикс = 2 мм по обеим осям).
        """
        return SceneParameters(
            height_mm=1000.0,
            width_mm=1000.0,
            height_px=500,
            width_px=500,
            light_x_mm=200.0,
            light_y_mm=-150.0,
            light_z_mm=500.0,
            axial_intensity_w_sr=100.0,
            circle_center_x_mm=0.0,
            circle_center_y_mm=0.0,
            circle_radius_mm=400.0,
        )


class ParameterValidator:
    """Проверка набора параметров сцены на соответствие ограничениям задачи."""

    @staticmethod
    def validate(parameters: SceneParameters) -> None:
        """Проверяет параметры и сообщает о первом найденном нарушении.

        Проверяются рекомендованные пределы всех величин, квадратность
        пикселя расчётной сетки и размещение расчётного круга целиком внутри
        области изображения.

        Args:
            parameters: Проверяемый набор исходных данных.

        Raises:
            ParameterValidationError: Если хотя бы одно ограничение нарушено.
        """
        ParameterValidator._check_range(
            parameters.height_mm, ParameterLimits.AREA_SIZE, "Высота области H"
        )
        ParameterValidator._check_range(
            parameters.width_mm, ParameterLimits.AREA_SIZE, "Ширина области W"
        )
        ParameterValidator._check_range(
            parameters.height_px, ParameterLimits.RESOLUTION, "Разрешение Hres"
        )
        ParameterValidator._check_range(
            parameters.width_px, ParameterLimits.RESOLUTION, "Разрешение Wres"
        )
        ParameterValidator._check_range(
            parameters.light_x_mm, ParameterLimits.LIGHT_XY, "Координата xL"
        )
        ParameterValidator._check_range(
            parameters.light_y_mm, ParameterLimits.LIGHT_XY, "Координата yL"
        )
        ParameterValidator._check_range(
            parameters.light_z_mm, ParameterLimits.LIGHT_Z, "Координата zL"
        )
        ParameterValidator._check_range(
            parameters.axial_intensity_w_sr,
            ParameterLimits.RADIANT_INTENSITY,
            "Сила излучения I0",
        )
        ParameterValidator._check_square_pixel(parameters)
        ParameterValidator._check_circle_fits(parameters)

    @staticmethod
    def _check_range(value: float, limits: ValueRange, title: str) -> None:
        """Проверяет одно скалярное значение по заданному диапазону.

        Args:
            value: Значение параметра.
            limits: Допустимый диапазон.
            title: Название параметра для сообщения об ошибке.

        Raises:
            ParameterValidationError: Если значение вне диапазона.
        """
        if not limits.contains(value):
            raise ParameterValidationError(
                f"{title}: значение {value:g} вне допустимых пределов "
                f"({limits.describe()})."
            )

    @staticmethod
    def _check_square_pixel(parameters: SceneParameters) -> None:
        """Проверяет условие квадратности пикселя H / Hres = W / Wres.

        Неквадратный пиксель искажает геометрию круга на изображении, поэтому
        постановка задачи требует равенства линейных размеров пикселя по осям.

        Raises:
            ParameterValidationError: Если размеры пикселя различаются больше
                допустимого относительного отклонения.
        """
        pixel_height = parameters.pixel_height_mm
        pixel_width = parameters.pixel_width_mm
        relative_difference = abs(pixel_height - pixel_width) / max(
            pixel_height, pixel_width
        )
        if relative_difference > SQUARE_PIXEL_RELATIVE_TOLERANCE:
            raise ParameterValidationError(
                "Пиксель должен быть квадратным: H / Hres = "
                f"{pixel_height:.4f} мм, W / Wres = {pixel_width:.4f} мм."
            )

    @staticmethod
    def _check_circle_fits(parameters: SceneParameters) -> None:
        """Проверяет радиус круга и его размещение внутри области изображения.

        Круг, выходящий за границы прямоугольника, нельзя рассчитать целиком,
        поэтому статистика по нему была бы неполной.

        Raises:
            ParameterValidationError: Если радиус неположителен или круг
                выходит за пределы области.
        """
        if parameters.circle_radius_mm <= 0.0:
            raise ParameterValidationError(
                "Радиус круга R должен быть положительным."
            )

        half_height = parameters.height_mm / 2.0
        half_width = parameters.width_mm / 2.0
        reaches_x = abs(parameters.circle_center_x_mm) + parameters.circle_radius_mm
        reaches_y = abs(parameters.circle_center_y_mm) + parameters.circle_radius_mm

        if reaches_x > half_height or reaches_y > half_width:
            raise ParameterValidationError(
                "Круг выходит за границы области изображения: требуется "
                f"|xc| + R <= H/2 ({reaches_x:g} <= {half_height:g}) и "
                f"|yc| + R <= W/2 ({reaches_y:g} <= {half_width:g})."
            )
