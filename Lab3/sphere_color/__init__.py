"""ЛР3: спектральное и RGB-моделирование освещения цветной сферы."""

from .models import LightSource, ParameterValidationError, ParameterValidator, SceneParameters, Spectrum
from .service import ComputationResult, LuminanceService

__all__ = [
    "LightSource", "ParameterValidationError", "ParameterValidator", "SceneParameters",
    "Spectrum", "ComputationResult", "LuminanceService",
]
