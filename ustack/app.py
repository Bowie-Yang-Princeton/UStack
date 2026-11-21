"""Start UStack: ``python -m ustack`` (or run_ustack.command / run_ustack.bat)."""

from __future__ import annotations

import os
import sys


def make_app():
    from .ui.style import FILE_OPENS, MAC, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("UStack")
    app.setOrganizationName("Uri Lab")
    if MAC:
        app.installEventFilter(FILE_OPENS)
    return app


def main() -> int:
    app = make_app()
    from .ui.style import ASSET_DIR, MAC, QtGui, install_mac_menu, setup_application
    from .ui.main_window import MainWindow
    setup_application(app)
    icon = ASSET_DIR / "UStack_icon_1024.png"
    if icon.is_file():
        app.setWindowIcon(QtGui.QIcon(str(icon)))
    window = MainWindow()
    screen = window.screen() or app.primaryScreen()
    if screen is not None:
        area = screen.availableGeometry()
        window.resize(min(window.width(), area.width() - 16), min(window.height(), area.height() - 16))
        window.move(area.center() - window.rect().center())
    install_mac_menu(window)
    window.show()
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json") and os.path.isfile(sys.argv[1]):
        from pathlib import Path
        window.open_device(Path(sys.argv[1]))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
