"""Запуск ЛР3: окно приложения или пакетное сравнение без интерфейса."""

from __future__ import annotations

import argparse
from pathlib import Path

from sphere_color import LuminanceService, SceneParameters


def main() -> None:
    parser = argparse.ArgumentParser(description="ЛР3: моделирование цветных изображений сферы")
    parser.add_argument("--batch", action="store_true", help="Сохранить оба варианта без GUI")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args()
    if args.batch:
        results = LuminanceService.compare_white_and_colored(SceneParameters.default())
        files = LuminanceService.save_comparison(results, args.output)
        print(f"Сохранено {len(files)} файлов в {args.output}")
    else:
        from sphere_color.ui import ApplicationWindow
        ApplicationWindow().mainloop()


if __name__ == "__main__":
    main()
