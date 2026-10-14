"""Seletor de ferramentas da barra lateral: botões em grade + painéis com barra de rolagem.

Substitui o QTabWidget: com 7 ferramentas a fileira de abas estourava a largura da barra.
Aqui todos os nomes ficam visíveis (em linhas), cada painel rola na vertical quando não cabe
e nunca precisa rolar na horizontal. Mesma interface básica do QTabWidget usada pela janela:
addTab, currentIndex, setCurrentIndex, tabText, count, currentChanged.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractButton, QButtonGroup, QComboBox, QFormLayout, QFrame, QGridLayout, QGroupBox,
    QLayout, QPushButton, QScrollArea, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

COLUMNS = 4


def compact(widget: QWidget) -> None:
    """Deixa os painéis mais enxutos: margens e espaços menores, formulários que quebram
    rótulos longos em vez de espremer os campos, listas suspensas que não alargam a barra."""
    lay = widget.layout()
    if lay is not None:
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)
    for gb in widget.findChildren(QGroupBox):
        if gb.layout() is not None:
            gb.layout().setContentsMargins(6, 4, 6, 6)
            gb.layout().setSpacing(4)
    for form in widget.findChildren(QFormLayout):
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setVerticalSpacing(4)
        form.setHorizontalSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setContentsMargins(0, 0, 0, 0)
    for lay2 in widget.findChildren(QLayout):
        if not isinstance(lay2, QFormLayout) and lay2 is not widget.layout() \
                and not isinstance(lay2.parent(), QGroupBox):
            lay2.setSpacing(4)
    for cb in widget.findChildren(QComboBox):
        cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        cb.setMinimumContentsLength(8)
        cb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


class ToolTabs(QWidget):
    currentChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 0)
        root.setSpacing(6)
        self._grid = QGridLayout()
        self._grid.setSpacing(4)
        root.addLayout(self._grid)
        self._stack = QStackedWidget()
        root.addWidget(self._stack, 1)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._titles: list[str] = []
        self._group.idClicked.connect(self.setCurrentIndex)

    def addTab(self, widget: QWidget, title: str, scroll: bool = True) -> int:
        """`scroll=False` para painéis que já têm a própria lista rolável (ex.: Filtros)."""
        compact(widget)
        idx = len(self._titles)
        if scroll:
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setFrameShape(QFrame.Shape.NoFrame)
            area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            area.setWidget(widget)
            page = area
        else:
            page = widget
        self._stack.addWidget(page)
        btn = QPushButton(title)
        btn.setObjectName("toolTab")
        btn.setCheckable(True)
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._group.addButton(btn, idx)
        self._grid.addWidget(btn, idx // COLUMNS, idx % COLUMNS)
        self._titles.append(title)
        if idx == 0:
            btn.setChecked(True)
        return idx

    def count(self) -> int:
        return len(self._titles)

    def tabText(self, index: int) -> str:
        return self._titles[index]

    def currentIndex(self) -> int:
        return self._stack.currentIndex()

    def setCurrentIndex(self, index: int) -> None:
        if not 0 <= index < self.count() or index == self._stack.currentIndex():
            return
        self._stack.setCurrentIndex(index)
        btn: QAbstractButton | None = self._group.button(index)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)
        self.currentChanged.emit(index)
