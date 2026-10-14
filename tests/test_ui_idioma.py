import os

import pytest

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_qt_standard_buttons_in_portuguese():
    from PyQt6.QtWidgets import QApplication, QDialogButtonBox, QMessageBox
    from eggfotos.ui.app import install_portuguese
    app = QApplication.instance() or QApplication([])
    app._tr = install_portuguese(app)
    mb = QMessageBox()
    for b in (QMessageBox.StandardButton.Save, QMessageBox.StandardButton.Discard,
              QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No):
        mb.addButton(b)
    textos = {b.text().replace("&", "") for b in mb.buttons()}
    assert textos == {"Salvar", "Descartar", "Cancelar", "Sim", "Não"}
    bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    assert {b.text() for b in bb.buttons()} == {"OK", "Cancelar"}
