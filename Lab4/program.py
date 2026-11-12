"""Запуск ЛР3: окно приложения или пакетное сравнение без интерфейса."""

from __future__ import annotations

import argparse
from pathlib import Path

from sphere_color import LuminanceService, SceneParameters


def main() -> None:
    parser = argparse.ArgumentParser(description="ЛР4: восстановление спектров из CIE RGB")
    parser.add_argument("--batch", action="store_true", help="Сохранить оба варианта без GUI")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--method", choices=("bounded", "basis"), default="bounded")
    parser.add_argument("--scene", type=Path, help="scene.json из предыдущего экспорта")
    parser.add_argument("--lab3-white", action="store_true", help="Материал исходной ЛР3, источники D65")
    args = parser.parse_args()
    if args.batch:
        from sphere_color.lab4_service import Lab4Service, default_scene
        from sphere_color.exporting import read_scene
        parameters = SceneParameters.default().with_white_lights() if args.lab3_white else default_scene()
        method, overrides = args.method, {}
        if args.scene:
            parameters, method, overrides = read_scene(args.scene)
        result = Lab4Service.compute(parameters, method, overrides)
        files = Lab4Service.save(result, args.output)
        print(f"Сохранено {len(files)} файлов в {args.output}")
    else:
        from sphere_color.lab4_ui import ApplicationWindow
        ApplicationWindow().mainloop()


if __name__ == "__main__":
    main()
