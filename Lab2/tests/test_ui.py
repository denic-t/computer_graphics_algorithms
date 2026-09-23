"""Проверки графического интерфейса через настоящие виджеты Tkinter.

Тесты работают с окном так же, как пользователь: вводят текст в поля,
нажимают кнопки и флажки, выбирают палитру. Всплывающие окна и диалог
сохранения подменяются, чтобы тесты не требовали участия человека.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Iterator
from pathlib import Path
from tkinter import ttk

import pytest
from PIL import Image

from sphere_luminance import ui
from sphere_luminance.models import MAXIMUM_LIGHT_COUNT, SceneParameters

# Время ожидания срабатывания автопересчёта с запасом на сам расчёт, мс.
SETTLE_MS = ui.AUTO_RECALCULATE_DELAY_MS + 400


class DialogRecorder:
    """Подмена messagebox: запоминает вызванные сообщения."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.infos: list[str] = []

    def showerror(self, title: str, message: str) -> None:
        self.errors.append(message)

    def showinfo(self, title: str, message: str) -> None:
        self.infos.append(message)


@pytest.fixture(scope="module")
def shared() -> Iterator[tuple[ui.ApplicationWindow, DialogRecorder]]:
    """Одно окно на модуль: повторное создание Tk() на Windows иногда
    приводит к сбою инициализации Tcl, поэтому окно переиспользуется."""
    recorder = DialogRecorder()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ui, "messagebox", recorder)
        application = ui.ApplicationWindow()
        application.update()
        yield application, recorder
        application.destroy()


@pytest.fixture()
def dialogs(shared: tuple[ui.ApplicationWindow, DialogRecorder]) -> DialogRecorder:
    return shared[1]


@pytest.fixture()
def window(shared: tuple[ui.ApplicationWindow, DialogRecorder]) -> ui.ApplicationWindow:
    """Окно в исходном состоянии: значения по умолчанию, автопересчёт, серая палитра."""
    application, recorder = shared
    application._auto_recalculate.set(True)
    application._colormap.set(ui.DISPLAY_COLORMAPS[0])
    application.reset_parameters()
    settle(application, 50)
    recorder.errors.clear()
    recorder.infos.clear()
    return application


def settle(window: tk.Tk, milliseconds: int = SETTLE_MS) -> None:
    """Прокручивает цикл событий заданное время (срабатывают after-задачи)."""
    window.after(milliseconds, window.quit)
    window.mainloop()


def all_widgets(root: tk.Misc) -> Iterator[tk.Misc]:
    for child in root.winfo_children():
        yield child
        yield from all_widgets(child)


def button(window: tk.Tk, text: str) -> ttk.Button:
    return next(w for w in all_widgets(window) if isinstance(w, ttk.Button) and w.cget("text") == text)


def entry(form: ui.NumberForm, label: str) -> ttk.Entry:
    """Поле ввода в строке формы с заданной подписью."""
    for widget in form.grid_slaves(column=0):
        if widget.cget("text") == label:
            row = int(widget.grid_info()["row"])
            return form.grid_slaves(row=row, column=1)[0]
    raise LookupError(label)


def type_into(field: ttk.Entry, text: str) -> None:
    field.delete(0, tk.END)
    field.insert(0, text)


def test_startup_shows_default_result(window: ui.ApplicationWindow) -> None:
    assert window._result is not None
    assert window._result.parameters == SceneParameters.default()
    assert window._status.get().startswith("Расчёт выполнен")
    report = window._report.get("1.0", tk.END)
    for title in ("Центр диска", "Максимум", "Минимум"):
        assert title in report


def test_typing_triggers_auto_recalculation(window: ui.ApplicationWindow) -> None:
    type_into(entry(window._material_form, "Блеск ke"), "20")
    settle(window)
    assert window._result.parameters.material.shininess == 20.0


def test_decimal_comma_is_accepted(window: ui.ApplicationWindow) -> None:
    type_into(entry(window._material_form, "Диффузное kd"), "0,45")
    settle(window)
    assert window._result.parameters.material.diffuse == pytest.approx(0.45)


def test_auto_recalculation_can_be_switched_off(window: ui.ApplicationWindow) -> None:
    next(w for w in all_widgets(window) if isinstance(w, ttk.Checkbutton) and w.cget("text") == "Автопересчёт").invoke()
    type_into(entry(window._material_form, "Блеск ke"), "20")
    settle(window)
    assert window._result.parameters.material.shininess == 80.0

    button(window, "Рассчитать (Enter)").invoke()
    assert window._result.parameters.material.shininess == 20.0


def test_invalid_input_goes_to_status_bar_only(
    window: ui.ApplicationWindow, dialogs: DialogRecorder
) -> None:
    previous = window._result
    type_into(entry(window._material_form, "Блеск ke"), "abc")
    settle(window)
    assert "не является числом" in window._status.get()
    assert dialogs.errors == []
    assert window._result is previous


def test_manual_recalculation_reports_error_dialog(
    window: ui.ApplicationWindow, dialogs: DialogRecorder
) -> None:
    type_into(entry(window._scene_form, "Разрешение Wres"), "500")
    button(window, "Рассчитать (Enter)").invoke()
    assert len(dialogs.errors) == 1
    assert "квадратным" in dialogs.errors[0]


def test_enter_key_recalculates(window: ui.ApplicationWindow) -> None:
    field = entry(window._scene_form, "Радиус сферы R")
    type_into(field, "250")
    field.focus_force()
    window.update()
    field.event_generate("<Return>")
    window.update()
    assert window._result.parameters.sphere_radius_mm == 250.0


def test_lights_can_be_added_until_limit_and_removed(window: ui.ApplicationWindow) -> None:
    add = button(window, "+ источник")
    while len(window._lights_panel._rows) < MAXIMUM_LIGHT_COUNT:
        add.invoke()
    assert add.instate(["disabled"])
    settle(window)
    assert len(window._result.parameters.lights) == MAXIMUM_LIGHT_COUNT

    remove_buttons = [w for w in all_widgets(window) if isinstance(w, ttk.Button) and w.cget("text") == "✕"]
    remove_buttons[0].invoke()
    assert not add.instate(["disabled"])
    settle(window)
    assert len(window._result.parameters.lights) == MAXIMUM_LIGHT_COUNT - 1


def test_light_checkbox_excludes_source(window: ui.ApplicationWindow, dialogs: DialogRecorder) -> None:
    checkboxes = [
        w for w in all_widgets(window._lights_panel) if isinstance(w, ttk.Checkbutton)
    ]
    checkboxes[1].invoke()
    settle(window)
    assert len(window._result.parameters.active_lights) == 1

    checkboxes[0].invoke()
    settle(window)
    assert "хотя бы один" in window._status.get()
    assert dialogs.errors == []


def test_colormap_changes_only_display(window: ui.ApplicationWindow) -> None:
    grey_before = window._result.grey_levels.copy()
    window._colormap.set("inferno")
    combobox = next(w for w in all_widgets(window) if isinstance(w, ttk.Combobox))
    combobox.event_generate("<<ComboboxSelected>>")
    window.update()
    assert window._canvas._image_axes.images[0].get_cmap().name == "inferno"
    assert (window._result.grey_levels == grey_before).all()


def test_save_writes_image_and_values(
    window: ui.ApplicationWindow,
    dialogs: DialogRecorder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "result.png"
    monkeypatch.setattr(ui.filedialog, "asksaveasfilename", lambda **_: str(target))
    button(window, "Сохранить изображение и значения").invoke()

    with Image.open(target) as image:
        assert image.mode == "L"
        assert image.size == (600, 600)
    values = (tmp_path / "result_values.txt").read_text(encoding="utf-8")
    assert "Максимум" in values and "Минимум" in values
    assert len(dialogs.infos) == 1


def test_reset_restores_defaults(window: ui.ApplicationWindow) -> None:
    button(window, "+ источник").invoke()
    type_into(entry(window._material_form, "Блеск ke"), "5")
    settle(window)
    button(window, "Значения по умолчанию").invoke()
    assert window._result.parameters == SceneParameters.default()
    assert entry(window._material_form, "Блеск ke").get() == "80"
