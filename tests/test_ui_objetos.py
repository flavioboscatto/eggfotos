"""Aba Fundo: escolha por tipo de objeto (manter/apagar) e ajuda automática da marca d'água."""
import os
import time

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import Qt

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TAB_FUNDO, TAB_MARCA = 4, 6
H, W = 240, 360


@pytest.fixture
def janela(qapp, fake_objects):
    from eggfotos.ui.main_window import MainWindow
    w = MainWindow()
    y, x = np.mgrid[0:H, 0:W]
    img = np.stack([x * 255 // W, y * 255 // H, (x + y) * 255 // (W + H)], 2).astype(np.uint8)
    w.history.reset(img)
    w._saved_uid = w.history.current_uid
    w._set_working_image(img, True)
    w._confirm_discard = lambda: True

    def esperar():
        t0 = time.time()
        while w._task is not None and time.time() - t0 < 20:
            qapp.processEvents()
        qapp.processEvents()
    w.esperar = esperar
    yield w
    w.close()


def test_reconhecer_lista_tipos_e_pessoas_ja_vem_para_manter(janela):
    w = janela
    w.tabs.setCurrentIndex(TAB_FUNDO)
    p = w.background_panel
    assert not p.btn_ouse.isEnabled() or p.btn_ouse.isHidden()   # nada reconhecido ainda
    p.btn_oscan.click(); w.esperar()
    assert set(p._obj_combos) == {"pessoas", "predios", "veiculos", "ceu"}
    assert p.objects_values()["manter"] == "pessoas" and p.objects_values()["apagar"] == ""
    assert not p.btn_ouse.isHidden()


def test_manter_pessoas_apagar_carros_e_aplicar_vira_passo(janela):
    w = janela
    w.tabs.setCurrentIndex(TAB_FUNDO)
    p = w.background_panel
    p.btn_oscan.click(); w.esperar()
    p._obj_combos["veiculos"].setCurrentIndex(2)           # Apagar
    p.btn_ouse.click(); w.esperar()
    assert w._matte is not None and w._matte_objects == {
        "manter": "pessoas", "apagar": "veiculos", "objetos_modelo": "rapido"}
    assert p.btn_apply.isEnabled()
    p.btn_apply.click(); w.esperar()
    step = w.history._entries[-1].step
    assert step["op"] == "remover_fundo" and step["manter"] == "pessoas" and step["apagar"] == "veiculos"
    out = w.image
    assert out.shape == (H, W, 4)
    assert out[150:200, 20:90, 3].min() == 255 and out[150:200, 260:340, 3].max() == 0


def test_so_apagar_sem_deteccao_principal_avisa(janela, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    avisos = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: avisos.append(a[2]))
    w = janela
    w.tabs.setCurrentIndex(TAB_FUNDO)
    p = w.background_panel
    p.btn_oscan.click(); w.esperar()
    p._obj_combos["pessoas"].setCurrentIndex(0)
    p._obj_combos["veiculos"].setCurrentIndex(2)
    p.btn_ouse.click(); w.esperar()
    assert avisos and "detecte" in avisos[0].lower()
    assert w._matte is None


def test_nada_escolhido_avisa(janela, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    avisos = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: avisos.append(a[2]))
    w = janela
    w.tabs.setCurrentIndex(TAB_FUNDO)
    p = w.background_panel
    p.btn_oscan.click(); w.esperar()
    p._obj_combos["pessoas"].setCurrentIndex(0)
    p.btn_ouse.click(); w.esperar()
    assert avisos and "Manter ou Apagar" in avisos[0]


def test_so_apagar_com_deteccao_principal(janela, fake_matte):
    w = janela
    w.tabs.setCurrentIndex(TAB_FUNDO)
    p = w.background_panel
    # a detecção "principal" falsa pega o quadrado central
    w._detect_background("equilibrado"); w.esperar()
    assert w._base_matte is not None
    p.btn_oscan.click(); w.esperar()
    p._obj_combos["pessoas"].setCurrentIndex(0)
    p._obj_combos["predios"].setCurrentIndex(2)             # apaga o prédio do meio
    p.btn_ouse.click(); w.esperar()
    assert w._matte_objects["manter"] == "" and w._matte_objects["apagar"] == "predios"
    meio = w._matte[110:130, 150:210]
    assert meio.max() < 30                                   # saiu da seleção


def test_trocar_de_foto_limpa_a_lista(janela):
    w = janela
    w.tabs.setCurrentIndex(TAB_FUNDO)
    p = w.background_panel
    p.btn_oscan.click(); w.esperar()
    assert p._obj_combos
    w.commit(w.image.copy(), "teste", None)
    assert not p._obj_combos and w._objmap is None and w._matte_objects is None


def test_clique_manual_descarta_tipos(janela):
    w = janela
    w._matte_objects = {"manter": "pessoas", "apagar": "", "objetos_modelo": "rapido"}
    w._sam_mask = np.full((H, W), 255, np.uint8)
    w._bg_click_use()
    assert w._matte_objects is None and w._matte_manual


# ---------------------------------------------- ajuda automática da marca d'água

def _foto_com_marca():
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:600, 0:900]
    base = np.stack([100 + 60 * y / 600, 150 + 50 * y / 600, 230 - 30 * y / 600], 2)
    base = np.clip(base + rng.standard_normal(base.shape), 0, 255).astype(np.uint8)
    ov = np.zeros(base.shape[:2], np.uint8)
    cv2.putText(ov, "WWW.EXEMPLO.COM", (560, 540), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
    a = (ov / 255.0 * 0.5)[..., None]
    return (base * (1 - a) + 255 * a).astype(np.uint8)


def _com_marca(w):
    img = _foto_com_marca()
    w.history.reset(img); w._saved_uid = w.history.current_uid
    w._set_working_image(img, True)
    w.tabs.setCurrentIndex(TAB_MARCA)


def test_procurar_lista_candidatas_sem_marcar_nada(janela):
    w = janela
    _com_marca(w)
    w.wm_panel.btn_find.click(); w.esperar()
    assert len(w._wm_cands) >= 1 and w.wm_panel.list_cand.count() == len(w._wm_cands)
    assert not w.wm_panel.g_cand.isHidden()
    assert w._wm_count() == 0 and not w.wm_panel.btn_apply.isEnabled()   # só aceitando é que vira marcação
    assert w.wm_panel.checked_candidates() == list(range(len(w._wm_cands)))


def test_aceitar_marcadas_vira_marcacao_e_desfaz_traz_a_candidata(janela):
    w = janela
    _com_marca(w)
    w.wm_panel.btn_find.click(); w.esperar()
    n = len(w._wm_cands)
    w.wm_panel.btn_cand_accept.click()
    assert w._wm_count() == n and not w._wm_cands and w.wm_panel.g_cand.isHidden()
    ys, xs = np.where(w._wm_mask > 0)
    assert xs.mean() > 600 and ys.mean() > 480             # canto inferior direito
    assert w.wm_panel.btn_apply.isEnabled()
    w.undo()
    assert w._wm_count() == n - 1 and len(w._wm_cands) == 1   # a candidata voltou para a lista


def test_desmarcar_e_aceitar_so_as_marcadas(janela):
    w = janela
    _com_marca(w)
    w._wm_cands = [_cand(0.05, 0.05, 0.2, 0.1), _cand(0.6, 0.8, 0.3, 0.1)]
    w._wm_sync_cands()
    w.wm_panel.list_cand.item(0).setCheckState(Qt.CheckState.Unchecked)
    w.wm_panel.btn_cand_accept.click()
    assert w._wm_count() == 1 and len(w._wm_cands) == 1
    assert w._wm_mask[:60, :90].max() == 0 and w._wm_mask[490:540, 560:860].max() == 255


def test_descartar_remove_da_lista_sem_marcar(janela):
    w = janela
    _com_marca(w)
    w._wm_cands = [_cand(0.05, 0.05, 0.2, 0.1), _cand(0.6, 0.8, 0.3, 0.1)]
    w._wm_sync_cands()
    w.wm_panel.list_cand.item(1).setCheckState(Qt.CheckState.Unchecked)
    w.wm_panel.btn_cand_discard.click()
    assert w._wm_count() == 0 and len(w._wm_cands) == 1 and w._wm_cands[0].x == 0.6   # descartou só a marcada (a 1)
    w.wm_panel.btn_cand_discard.click()
    assert not w._wm_cands and w.wm_panel.g_cand.isHidden()


def test_aceitar_so_uma_parte(janela):
    w = janela
    _com_marca(w)
    w._wm_cands = [_cand(0.5, 0.8, 0.4, 0.1)]            # x 450–810, y 480–540 (foto 900×600)
    w._wm_sync_cands()
    w.wm_panel.btn_cand_partial.click()
    assert w._wm_partial == [0] and not w.wm_panel.btn_cand_partial.isEnabled()
    w._apply_region_blur((600, 470, 100, 100))            # retângulo cobre só um pedaço
    assert w._wm_partial is None and w.wm_panel.btn_cand_partial.isEnabled()
    assert w._wm_count() == 1 and len(w._wm_cands) == 1   # a candidata continua listada
    ys, xs = np.where(w._wm_mask > 0)
    assert xs.min() == 600 and xs.max() == 699 and ys.min() == 480 and ys.max() == 539


def test_parte_fora_da_candidata_nao_marca_nada(janela):
    w = janela
    _com_marca(w)
    w._wm_cands = [_cand(0.5, 0.8, 0.4, 0.1)]
    w._wm_sync_cands()
    w.wm_panel.btn_cand_partial.click()
    w._apply_region_blur((10, 10, 50, 50))
    assert w._wm_count() == 0 and w._wm_partial is None


def test_mexer_na_ferramenta_cancela_aceitar_parte(janela):
    w = janela
    _com_marca(w)
    w._wm_cands = [_cand(0.5, 0.8, 0.4, 0.1)]
    w._wm_sync_cands()
    w.wm_panel.btn_cand_partial.click()
    w.wm_panel.grp_tool.button(0).setChecked(False)
    w.wm_panel.grp_tool.button(1).setChecked(True)
    assert w._wm_partial is None


def test_nova_foto_limpa_candidatas(janela):
    w = janela
    _com_marca(w)
    w.wm_panel.btn_find.click(); w.esperar()
    assert w._wm_cands
    w.commit(w.image.copy(), "teste", None)
    assert not w._wm_cands and w.wm_panel.g_cand.isHidden()


def _cand(x, y, w, h):
    from eggfotos.core.suggest import Candidate
    return Candidate(x, y, w, h, 0.5, "clara", "canto", "texto")


def test_sugerir_ajustes_preenche_os_controles(janela):
    w = janela
    img = _foto_com_marca()
    w.history.reset(img); w._saved_uid = w.history.current_uid
    w._set_working_image(img, True)
    w.tabs.setCurrentIndex(TAB_MARCA)
    assert not w.wm_panel.btn_suggest.isEnabled()
    w._apply_region_blur((540, 490, 340, 70)); w.esperar()
    assert w.wm_panel.btn_suggest.isEnabled()
    w.wm_panel.btn_suggest.click(); w.esperar()
    assert w.wm_panel.metodo() == "telea"                    # fundo liso
    assert not w.wm_panel.lbl_sugestao.isHidden() and "liso" in w.wm_panel.lbl_sugestao.text()
    assert w.wm_panel.cmb_pol.currentData() == "clara"
