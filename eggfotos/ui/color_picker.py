"""Seletor de cor com paleta: amostras prontas + "Mais cores…" (roda de cores e RGB)."""
from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog, QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout,
)

PALETTE = (
    "#000000", "#404040", "#808080", "#c0c0c0", "#ffffff", "#7f0000",
    "#e53935", "#fb8c00", "#fdd835", "#43a047", "#00897b", "#039be5",
    "#1e3a8a", "#5e35b1", "#d81b60", "#8d6e63", "#f5deb3", "#ffb6c1",
)
COLUMNS = 6


def style_color_button(btn: QPushButton, color: str) -> None:
    """Pinta o botão com a cor e mostra o valor RGB, com texto legível sobre ela."""
    c = QColor(color)
    fg = "#000000" if c.lightness() > 140 else "#ffffff"
    btn.setText(color.upper())
    btn.setToolTip(f"RGB {c.red()}, {c.green()}, {c.blue()} — clique para escolher outra cor")
    btn.setStyleSheet(f"background: {color}; color: {fg};")


class ColorPaletteDialog(QDialog):
    def __init__(self, current: str, title: str = "Escolher cor", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._color = QColor(current) if QColor(current).isValid() else QColor("#000000")

        lay = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setSpacing(6)
        for i, hex_ in enumerate(PALETTE):
            b = QPushButton()
            b.setFixedSize(40, 32)
            b.setToolTip(hex_.upper())
            b.setStyleSheet(f"background: {hex_}; border: 1px solid #555; border-radius: 4px;")
            b.clicked.connect(lambda _=False, h=hex_: self._set(h))
            grid.addWidget(b, i // COLUMNS, i % COLUMNS)
        lay.addLayout(grid)

        row = QHBoxLayout()
        self.preview = QLabel()
        self.preview.setFixedSize(40, 32)
        self.lbl = QLabel()
        more = QPushButton("Mais cores…")
        more.clicked.connect(self._more)
        row.addWidget(self.preview)
        row.addWidget(self.lbl, 1)
        row.addWidget(more)
        lay.addLayout(row)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._refresh()

    def _set(self, hex_: str) -> None:
        self._color = QColor(hex_)
        self._refresh()

    def _refresh(self) -> None:
        c = self._color
        self.preview.setStyleSheet(f"background: {c.name()}; border: 1px solid #555; border-radius: 4px;")
        self.lbl.setText(f"{c.name().upper()}   RGB {c.red()}, {c.green()}, {c.blue()}")

    def _more(self) -> None:
        # janela própria do Qt (não a do macOS): abre sempre, com roda de cores e campos RGB
        c = QColorDialog.getColor(self._color, self, "Mais cores",
                                  QColorDialog.ColorDialogOption.DontUseNativeDialog)
        if c.isValid():
            self._color = c
            self._refresh()

    def color(self) -> str:
        return self._color.name()


def choose_color(parent, current: str, title: str = "Escolher cor") -> str | None:
    """Abre a paleta. Devolve '#rrggbb' ou None se cancelado."""
    dlg = ColorPaletteDialog(current, title, parent)
    return dlg.color() if dlg.exec() else None
