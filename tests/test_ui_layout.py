import os

import pytest

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_tool_tabs_basic(app):
    from PyQt6.QtWidgets import QLabel, QScrollArea
    from eggfotos.ui.tool_tabs import ToolTabs
    t = ToolTabs()
    seen = []
    t.currentChanged.connect(seen.append)
    for i in range(7):
        t.addTab(QLabel(f"p{i}"), f"Aba {i}", scroll=(i != 1))
    assert t.count() == 7 and t.currentIndex() == 0 and t.tabText(6) == "Aba 6"
    t.setCurrentIndex(3)
    assert t.currentIndex() == 3 and seen == [3]
    t.setCurrentIndex(3)                       # repetir não emite de novo
    t.setCurrentIndex(99)                      # fora da faixa é ignorado
    assert seen == [3]
    assert isinstance(t._stack.widget(0), QScrollArea) and not isinstance(t._stack.widget(1), QScrollArea)


def test_sidebar_has_no_horizontal_overflow_and_remembers_width(app):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QScrollArea
    from eggfotos.ui.main_window import MainWindow
    from eggfotos.ui.styles import STYLESHEET
    app.setStyleSheet(STYLESHEET)
    w = MainWindow()
    w.settings.clear()
    w.resize(1300, 800)
    w.show()
    for i in range(w.tabs.count()):
        w.tabs.setCurrentIndex(i)
        page = w.tabs._stack.currentWidget()
        inner = page.widget() if isinstance(page, QScrollArea) else page
        viewport = page.viewport().width() if isinstance(page, QScrollArea) else page.width()
        assert inner.minimumSizeHint().width() <= viewport, w.tabs.tabText(i)   # nada estoura na largura
    w.resizeDocks([w.dock_tools], [470], Qt.Orientation.Horizontal)
    app.processEvents()
    largura = w.dock_tools.width()
    w._confirm_discard = lambda: True
    w.close()
    w2 = MainWindow()
    w2.resize(1300, 800)
    w2.show()
    app.processEvents()
    assert abs(w2.dock_tools.width() - largura) <= 12
    w2.settings.clear()
    w2.close()


def test_watermark_panel_controls_follow_the_method(app):
    from eggfotos.ui.panels import WatermarkPanel
    p = WatermarkPanel()
    p.show()
    p.cmb_metodo.setCurrentIndex(p.cmb_metodo.findData("telea"))
    assert p.sl_raio.isVisible() and not p.sl_contexto.isVisible()
    p.cmb_metodo.setCurrentIndex(p.cmb_metodo.findData("lama"))
    assert p.sl_contexto.isVisible() and not p.sl_raio.isVisible()
    for sl in (p.sl_expandir, p.sl_contexto, p.sl_emenda, p.sl_grao):
        assert sl.isEnabled() and sl.isVisible()          # tudo configurável com o LaMa
    p.sl_grao.setValue(0)
    v = p.values()
    assert v["grao"] is False and v["grao_forca"] == 0 and v["metodo"] == "lama" and "contexto" in v
