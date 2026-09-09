"""Пакет расчёта и визуализации освещённости плоскости.

Разделение ответственности между модулями:
    * models    — исходные данные сцены и проверка их допустимости;
    * physics   — расчёт освещённости точечным ламбертовским источником;
    * analysis  — контрольные значения, статистика и сечения распределения;
    * rendering — нормировка к градациям серого и запись изображения в файл;
    * service   — сценарий расчёта целиком;
    * ui        — графический интерфейс пользователя на Tkinter.
"""

from .models import (
    ParameterLimits,
    ParameterValidationError,
    ParameterValidator,
    SceneParameters,
)
from .physics import IlluminanceCalculator, IlluminanceField
from .service import ComputationResult, IlluminanceService

__all__ = [
    "ComputationResult",
    "IlluminanceCalculator",
    "IlluminanceField",
    "IlluminanceService",
    "ParameterLimits",
    "ParameterValidationError",
    "ParameterValidator",
    "SceneParameters",
]
