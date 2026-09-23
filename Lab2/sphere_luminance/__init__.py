"""Пакет расчёта и визуализации яркости на сфере от точечных источников света.

Разделение ответственности между модулями:
    * models    — исходные данные сцены и проверка их допустимости;
    * geometry  — лучи зрения через пиксели экрана и пересечение со сферой;
    * physics   — яркость поверхности по модели Блинн-Фонга;
    * analysis  — контрольные точки, статистика и сечения распределения;
    * rendering — нормировка к градациям серого, запись изображения и отчёта;
    * service   — сценарий расчёта целиком;
    * ui        — графический интерфейс пользователя на Tkinter.
"""

from .models import (
    BlinnPhongMaterial,
    LightSource,
    ParameterLimits,
    ParameterValidationError,
    ParameterValidator,
    SceneParameters,
)
from .physics import LuminanceCalculator, LuminanceField
from .service import ComputationResult, LuminanceService

__all__ = [
    "BlinnPhongMaterial",
    "ComputationResult",
    "LightSource",
    "LuminanceCalculator",
    "LuminanceField",
    "LuminanceService",
    "ParameterLimits",
    "ParameterValidationError",
    "ParameterValidator",
    "SceneParameters",
]
