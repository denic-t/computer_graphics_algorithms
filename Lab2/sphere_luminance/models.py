"""Описание параметров сцены и проверка их допустимости.

Модуль отвечает только за структуру входных данных задачи и за контроль
рекомендованных пределов. Расчётов яркости здесь нет.

Принятая система координат (соответствует рисунку из постановки задачи):

    * прямоугольный экран размером H x W [мм] лежит в плоскости z = 0,
      начало координат расположено в его центре;
    * ось X направлена вдоль высоты экрана H и отображается по вертикали
      изображения (Hres пикселей, верхняя строка соответствует x = +H/2);
    * ось Y направлена вдоль ширины экрана W и отображается по горизонтали
      изображения (Wres пикселей);
    * ось Z направлена вверх, к наблюдателю O(0, 0, zO) и источникам света;
    * наблюдатель смотрит на экран, поэтому видимая область — пирамида с
      вершиной O и основанием-экраном; сфера должна целиком лежать внутри неё.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Относительный допуск при проверке квадратности пикселя (как в ЛР1):
# пользователь вводит целые пиксели и миллиметры, точное равенство
# размеров пикселя по осям недостижимо.
SQUARE_PIXEL_RELATIVE_TOLERANCE = 1e-3

# Ограничение числа источников: интерфейс остаётся обозримым, а расчёт —
# быстрым даже при максимальном разрешении 800 x 800.
MAXIMUM_LIGHT_COUNT = 6


class ParameterValidationError(ValueError):
    """Ошибка недопустимого значения параметра сцены.

    Отдельный тип исключения позволяет интерфейсу пользователя отличать
    ошибки ввода от программных сбоев и показывать их как сообщение.
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
        """Проверяет попадание значения в диапазон."""
        return self.minimum <= value <= self.maximum

    def describe(self) -> str:
        """Возвращает текстовое описание диапазона для сообщений об ошибке."""
        return f"от {self.minimum:g} до {self.maximum:g} {self.unit}".rstrip()


class ParameterLimits:
    """Рекомендованные пределы значений параметров из постановки задачи.

    Пределы для zO и коэффициентов модели Блинн-Фонга в задании не заданы;
    они выбраны так, чтобы отсечь заведомо бессмысленный ввод.
    """

    SCREEN_SIZE = ValueRange(100.0, 10000.0, "мм")
    RESOLUTION = ValueRange(200, 800, "пикс")
    OBSERVER_Z = ValueRange(100.0, 100000.0, "мм")
    LIGHT_XY = ValueRange(-10000.0, 10000.0, "мм")
    LIGHT_Z = ValueRange(100.0, 10000.0, "мм")
    RADIANT_INTENSITY = ValueRange(0.01, 10000.0, "Вт/ср")
    SPHERE_XY = ValueRange(-10000.0, 10000.0, "мм")
    SPHERE_Z = ValueRange(100.0, 10000.0, "мм")
    REFLECTION_COEFFICIENT = ValueRange(0.0, 10.0, "")
    SHININESS = ValueRange(1.0, 10000.0, "")


@dataclass(frozen=True)
class LightSource:
    """Точечный источник света с ламбертовской диаграммой излучения.

    Ось диаграммы направлена вертикально вниз (0, 0, -1), поэтому сила
    излучения в направлении под углом theta к оси равна I0 * cos(theta).

    Attributes:
        x_mm: Координата xL источника, мм.
        y_mm: Координата yL источника, мм.
        z_mm: Координата zL источника, мм.
        intensity_w_sr: Сила излучения I0 вдоль оси диаграммы, Вт/ср.
        enabled: False — источник временно исключён из расчёта.
    """

    x_mm: float
    y_mm: float
    z_mm: float
    intensity_w_sr: float
    enabled: bool = True

    @property
    def position_mm(self) -> tuple[float, float, float]:
        """Координаты источника (xL, yL, zL), мм."""
        return self.x_mm, self.y_mm, self.z_mm


@dataclass(frozen=True)
class BlinnPhongMaterial:
    """Свойства поверхности сферы по модели Блинн-Фонга.

    Двунаправленная функция отражения f = kd + ks * (h . N)^ke.

    Attributes:
        diffuse: Коэффициент диффузного отражения kd.
        specular: Коэффициент зеркального отражения ks.
        shininess: Показатель блеска ke: чем больше, тем уже блик.
    """

    diffuse: float
    specular: float
    shininess: float


@dataclass(frozen=True)
class SceneParameters:
    """Полный набор исходных данных для расчёта яркости на сфере.

    Attributes:
        height_mm: Высота экрана H вдоль оси X, мм.
        width_mm: Ширина экрана W вдоль оси Y, мм.
        height_px: Разрешение изображения по высоте Hres, пикселей.
        width_px: Разрешение изображения по ширине Wres, пикселей.
        observer_z_mm: Высота наблюдателя zO над экраном, мм.
        sphere_x_mm: Координата xC центра сферы, мм.
        sphere_y_mm: Координата yC центра сферы, мм.
        sphere_z_mm: Координата zC центра сферы, мм.
        sphere_radius_mm: Радиус сферы R, мм.
        material: Параметры модели Блинн-Фонга.
        lights: Точечные источники света.
    """

    height_mm: float
    width_mm: float
    height_px: int
    width_px: int
    observer_z_mm: float
    sphere_x_mm: float
    sphere_y_mm: float
    sphere_z_mm: float
    sphere_radius_mm: float
    material: BlinnPhongMaterial
    lights: tuple[LightSource, ...] = field(default_factory=tuple)

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
        """Границы экрана по оси X: (-H/2, +H/2), мм."""
        return -self.height_mm / 2.0, self.height_mm / 2.0

    @property
    def y_bounds_mm(self) -> tuple[float, float]:
        """Границы экрана по оси Y: (-W/2, +W/2), мм."""
        return -self.width_mm / 2.0, self.width_mm / 2.0

    @property
    def observer_mm(self) -> tuple[float, float, float]:
        """Координаты наблюдателя (0, 0, zO), мм."""
        return 0.0, 0.0, self.observer_z_mm

    @property
    def sphere_center_mm(self) -> tuple[float, float, float]:
        """Координаты центра сферы (xC, yC, zC), мм."""
        return self.sphere_x_mm, self.sphere_y_mm, self.sphere_z_mm

    @property
    def active_lights(self) -> tuple[LightSource, ...]:
        """Источники, участвующие в расчёте."""
        return tuple(light for light in self.lights if light.enabled)

    @staticmethod
    def default() -> "SceneParameters":
        """Возвращает корректный набор параметров, используемый при запуске.

        Экран 1000 x 1000 мм при 600 x 600 пикс даёт квадратный пиксель.
        Два источника по разные стороны от сферы создают два блика, а
        ke = 80 делает пик зеркального отражения ярко выраженным.
        """
        return SceneParameters(
            height_mm=1000.0,
            width_mm=1000.0,
            height_px=600,
            width_px=600,
            observer_z_mm=3000.0,
            sphere_x_mm=0.0,
            sphere_y_mm=0.0,
            sphere_z_mm=400.0,
            sphere_radius_mm=300.0,
            material=BlinnPhongMaterial(diffuse=0.3, specular=0.7, shininess=80.0),
            lights=(
                LightSource(x_mm=-900.0, y_mm=700.0, z_mm=2000.0, intensity_w_sr=1000.0),
                LightSource(x_mm=800.0, y_mm=-900.0, z_mm=1200.0, intensity_w_sr=500.0),
            ),
        )


class ParameterValidator:
    """Проверка набора параметров сцены на соответствие ограничениям задачи."""

    @staticmethod
    def validate(parameters: SceneParameters) -> None:
        """Проверяет параметры и сообщает о первом найденном нарушении.

        Args:
            parameters: Проверяемый набор исходных данных.

        Raises:
            ParameterValidationError: Если хотя бы одно ограничение нарушено.
        """
        check = ParameterValidator._check_range
        limits = ParameterLimits

        check(parameters.height_mm, limits.SCREEN_SIZE, "Высота экрана H")
        check(parameters.width_mm, limits.SCREEN_SIZE, "Ширина экрана W")
        check(parameters.height_px, limits.RESOLUTION, "Разрешение Hres")
        check(parameters.width_px, limits.RESOLUTION, "Разрешение Wres")
        check(parameters.observer_z_mm, limits.OBSERVER_Z, "Наблюдатель zO")
        check(parameters.sphere_x_mm, limits.SPHERE_XY, "Центр сферы xC")
        check(parameters.sphere_y_mm, limits.SPHERE_XY, "Центр сферы yC")
        check(parameters.sphere_z_mm, limits.SPHERE_Z, "Центр сферы zC")

        material = parameters.material
        check(material.diffuse, limits.REFLECTION_COEFFICIENT, "Коэффициент kd")
        check(material.specular, limits.REFLECTION_COEFFICIENT, "Коэффициент ks")
        check(material.shininess, limits.SHININESS, "Показатель ke")

        ParameterValidator._check_lights(parameters.lights)
        ParameterValidator._check_square_pixel(parameters)
        ParameterValidator._check_sphere_visible(parameters)

    @staticmethod
    def _check_range(value: float, limits: ValueRange, title: str) -> None:
        """Проверяет одно скалярное значение по заданному диапазону.

        Raises:
            ParameterValidationError: Если значение вне диапазона.
        """
        if not limits.contains(value):
            raise ParameterValidationError(
                f"{title}: значение {value:g} вне допустимых пределов "
                f"({limits.describe()})."
            )

    @staticmethod
    def _check_lights(lights: tuple[LightSource, ...]) -> None:
        """Проверяет количество источников и пределы их параметров.

        Raises:
            ParameterValidationError: Если нет ни одного включённого источника,
                их слишком много или параметры источника вне пределов.
        """
        if len(lights) > MAXIMUM_LIGHT_COUNT:
            raise ParameterValidationError(
                f"Допускается не более {MAXIMUM_LIGHT_COUNT} источников света."
            )
        if not any(light.enabled for light in lights):
            raise ParameterValidationError(
                "Должен быть включён хотя бы один источник света."
            )

        for number, light in enumerate(lights, start=1):
            prefix = f"Источник {number}"
            check = ParameterValidator._check_range
            check(light.x_mm, ParameterLimits.LIGHT_XY, f"{prefix}: xL")
            check(light.y_mm, ParameterLimits.LIGHT_XY, f"{prefix}: yL")
            check(light.z_mm, ParameterLimits.LIGHT_Z, f"{prefix}: zL")
            check(light.intensity_w_sr, ParameterLimits.RADIANT_INTENSITY, f"{prefix}: I0")

    @staticmethod
    def _check_square_pixel(parameters: SceneParameters) -> None:
        """Проверяет условие квадратности пикселя H / Hres = W / Wres.

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
    def _check_sphere_visible(parameters: SceneParameters) -> None:
        """Проверяет, что сфера целиком лежит внутри пирамиды видимости.

        Пирамида ограничена плоскостью экрана z = 0 и четырьмя гранями,
        проходящими через наблюдателя O(0, 0, zO) и края экрана. Грань,
        проходящая через край x = a (a = H/2), задаётся уравнением

            x + a * z / zO - a = 0,  нормаль n = (1, 0, a / zO).

        Центр C лежит внутри не ближе R к грани, если знаковое расстояние
        (|xC| + a * zC / zO - a) / |n| <= -R. Для граней y = ±W/2 — аналогично.

        Raises:
            ParameterValidationError: Если радиус неположителен или сфера
                выходит за пределы видимой области.
        """
        radius = parameters.sphere_radius_mm
        if radius <= 0.0:
            raise ParameterValidationError("Радиус сферы R должен быть положительным.")

        observer_z = parameters.observer_z_mm
        center_z = parameters.sphere_z_mm

        if center_z - radius < 0.0:
            raise ParameterValidationError(
                "Сфера пересекает плоскость экрана: требуется zC - R >= 0 "
                f"({center_z - radius:g} мм)."
            )
        if center_z + radius >= observer_z:
            raise ParameterValidationError(
                "Наблюдатель должен находиться выше сферы: требуется "
                f"zC + R < zO ({center_z + radius:g} >= {observer_z:g})."
            )

        faces = (
            ("x", abs(parameters.sphere_x_mm), parameters.height_mm / 2.0),
            ("y", abs(parameters.sphere_y_mm), parameters.width_mm / 2.0),
        )
        for axis, center_offset, half_size in faces:
            slope = half_size / observer_z
            signed_distance = (center_offset + slope * center_z - half_size) / math.hypot(
                1.0, slope
            )
            if signed_distance > -radius:
                raise ParameterValidationError(
                    f"Сфера выходит за пирамиду видимости по оси {axis.upper()}: "
                    f"расстояние от центра до грани {-signed_distance:.1f} мм "
                    f"меньше радиуса R = {radius:g} мм."
                )
