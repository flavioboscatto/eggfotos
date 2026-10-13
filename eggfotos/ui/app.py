"""Inicialização da aplicação Qt."""
from __future__ import annotations

import sys
import traceback

from PyQt6.QtWidgets import QApplication, QMessageBox

from .. import APP_NAME
from .main_window import MainWindow
from .styles import STYLESHEET


def _install_excepthook() -> None:
    """Mostra erros não tratados numa caixa de diálogo em vez de fechar o app
    silenciosamente (útil durante os testes com o Claude Code)."""
    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        print(text, file=sys.stderr)
        app = QApplication.instance()
        if app is not None:
            box = QMessageBox(QMessageBox.Icon.Critical, APP_NAME, f"Erro inesperado:\n{exc}")
            box.setDetailedText(text)
            box.exec()
    sys.excepthook = hook


def run(argv: list[str]) -> int:
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    _install_excepthook()
    win = MainWindow()
    win.show()
    if len(argv) > 1:  # permite "Abrir com…" e arrastar para o executável
        win.open_file(argv[1])
    return app.exec()
