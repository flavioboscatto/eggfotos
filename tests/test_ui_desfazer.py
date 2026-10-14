"""Ctrl+Z / Ctrl+Y passo a passo nas marcações (aba Marca d'água e cliques do Fundo)."""
import os
import time

import numpy as np
import pytest

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TAB_FUNDO, TAB_MARCA = 4, 6


@pytest.fixture
def janela(monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from eggfotos.ui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    img = np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    w.history.reset(img)
    w._saved_uid = w.history.current_uid
    w._set_working_image(img, True)
    w._confirm_discard = lambda: True

    def esperar():
        t0 = time.time()
        while w._task is not None and time.time() - t0 < 20:
            app.processEvents()
        app.processEvents()
    w.esperar = esperar
    yield w
    w.close()


def _px(w):
    return int((w._wm_mask > 0).sum()) if w._wm_mask is not None else 0


def test_marcacoes_desfazem_uma_por_vez(janela):
    w = janela
    w.tabs.setCurrentIndex(TAB_MARCA)
    assert not w.act_undo.isEnabled()
    w._apply_region_blur((10, 10, 50, 40)); a = _px(w)
    w._apply_region_blur((100, 10, 60, 40)); b = _px(w)
    traco = np.zeros((300, 400), np.uint8); traco[200:220, 20:150] = 255
    w._apply_brush_blur(traco); c = _px(w)
    assert a < b < c and w.act_undo.isEnabled() and "marcação" in w.act_undo.text()
    w.undo(); assert _px(w) == b
    w.undo(); assert _px(w) == a
    w.redo(); assert _px(w) == b
    w.undo(); w.undo(); assert _px(w) == 0
    assert len(w.history._entries) == 1          # o histórico da foto não foi tocado
    assert not w.act_undo.isEnabled()


def test_limpar_e_apertar_tambem_desfazem(janela):
    w = janela
    w.tabs.setCurrentIndex(TAB_MARCA)
    w._apply_region_blur((10, 10, 50, 40)); n = _px(w)
    w.wm_panel.btn_clear.click(); assert _px(w) == 0
    w.undo(); assert _px(w) == n


def test_desfazer_remocao_devolve_as_marcacoes(janela):
    w = janela
    w.tabs.setCurrentIndex(TAB_MARCA)
    w._apply_region_blur((10, 10, 50, 40)); n = _px(w)
    w.wm_panel.btn_apply.click(); w.esperar()
    assert len(w.history._entries) == 2 and _px(w) == 0
    w.undo()                                     # desfaz a remoção…
    assert len(w.history._entries) == 2 and w.history.can_redo()
    assert _px(w) == n                           # …e as marcações voltam para ajustar


def test_cliques_do_fundo_desfazem_um_a_um(janela, fake_sam, monkeypatch):
    from eggfotos.core import segment
    monkeypatch.setattr(segment, "model_available", lambda: True)
    w = janela
    w.tabs.setCurrentIndex(TAB_FUNDO)
    w._bg_click_start()
    w._on_point_clicked(100, 100, True); w.esperar()
    w._on_point_clicked(300, 200, True); w.esperar()
    assert len(w._sam_points) == 2 and "marcação" in w.act_undo.text()
    w.undo(); w.esperar()
    assert len(w._sam_points) == 1 and len(w.canvas._point_items) == 1
    w.redo(); w.esperar()
    assert len(w._sam_points) == 2
    w.undo(); w.undo(); w.esperar()
    assert w._sam_points == [] and w._sam_mask is None
    assert len(w.history._entries) == 1
