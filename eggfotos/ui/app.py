"""Inicialização da aplicação Qt."""
from __future__ import annotations

import sys
import traceback

from PyQt6.QtCore import QLibraryInfo, QLocale, QTranslator
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMessageBox

from .. import APP_NAME, ICON_PATH
from ..errors import friendly_error
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
            box = QMessageBox(QMessageBox.Icon.Critical, APP_NAME, f"Erro inesperado:\n{friendly_error(exc)}")
            box.setDetailedText(text)
            box.setStandardButtons(QMessageBox.StandardButton.Ok)
            box.exec()
    sys.excepthook = hook


def install_portuguese(app: QApplication) -> list[QTranslator]:
    """Botões e diálogos padrão do Qt (Sim/Não, Salvar, Cancelar, seletor de cores…) em português
    do Brasil, e números com vírgula decimal. Devolve os tradutores (precisam ficar referenciados)."""
    QLocale.setDefault(QLocale(QLocale.Language.Portuguese, QLocale.Country.Brazil))
    folder = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    translators = []
    for name in ("qtbase_pt_BR", "qt_pt_BR"):
        t = QTranslator(app)
        if t.load(name, folder):
            app.installTranslator(t)
            translators.append(t)
    return translators


def run(argv: list[str]) -> int:
    app = QApplication(argv)
    app._translators = install_portuguese(app)  # noqa: SLF001 — mantém os tradutores vivos
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    if ICON_PATH.is_file():
        app.setWindowIcon(QIcon(str(ICON_PATH)))  # janela e Dock
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    _install_excepthook()
    win = MainWindow()
    win.show()
    if len(argv) > 1:  # permite "Abrir com…" e arrastar para o executável
        win.open_file(argv[1])
    return app.exec()
