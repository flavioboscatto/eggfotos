import os

import pytest

from eggfotos.core import presets
from eggfotos.core.inpaint import METHODS
from eggfotos.pipeline import operations as ops


def test_every_preset_is_valid_and_usable_in_the_pipeline():
    assert len(presets.PRESETS) >= 8
    for key, p in presets.PRESETS.items():
        v = p.values()
        assert p.metodo in METHODS and p.dica and p.label, key
        for campo, (lo, hi) in presets.RANGES.items():
            assert lo <= v[campo] <= hi, (key, campo)
        step = ops.get_operation("remover_marca_regiao").new_step(
            x=0.1, y=0.1, w=0.2, h=0.1, metodo=v["metodo"], expandir=v["expandir"], raio=v["raio"],
            contexto=v["contexto"], emenda=v["emenda"], grao_forca=v["grao_forca"])
        assert ops.normalize_step(step)["metodo"] == v["metodo"]            # a receita aceita o perfil


def test_presets_differ_where_the_advice_says_so():
    P = presets.PRESETS
    assert P["liso"].metodo == "telea" and P["textura"].metodo == "lama"
    assert P["detalhes"].contexto > P["textura"].contexto > P["grande"].contexto
    assert P["grao_alto"].grao_forca > 100 > P["foto_limpa"].grao_forca
    assert P["retangulo"].expandir < P["texto_fino"].expandir


def test_user_profiles_roundtrip(marks_tmp):
    assert presets.load_user_profiles() == {}
    presets.save_user_profile("Meu céu", {"metodo": "telea", "expandir": 20, "raio": 70, "contexto": 40,
                                          "emenda": 15, "grao_forca": 90, "sensibilidade": 55})
    presets.save_user_profile("Outro", {"metodo": "lama", "expandir": 999, "grao_forca": -5})
    perfis = presets.load_user_profiles()
    assert set(perfis) == {"Meu céu", "Outro"} and perfis["Meu céu"]["raio"] == 70
    assert perfis["Outro"]["expandir"] == 100 and perfis["Outro"]["grao_forca"] == 0   # limitado à faixa
    presets.delete_user_profile("Outro")
    assert set(presets.load_user_profiles()) == {"Meu céu"}
    presets.delete_user_profile("nao existe")


def test_user_profile_validation(marks_tmp):
    for nome in ("", "  ", presets.CUSTOM, presets.PRESETS["liso"].label):
        with pytest.raises(ValueError):
            presets.save_user_profile(nome, {"metodo": "telea"})
    with pytest.raises(ValueError):
        presets.save_user_profile("x", {"metodo": "inexistente"})


def test_corrupt_profiles_file_is_ignored(marks_tmp):
    (marks_tmp / "perfis.json").write_text("{ não é json", encoding="utf-8")
    assert presets.load_user_profiles() == {}


# ------------------------------------------------------------------ painel
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def painel(marks_tmp, qapp):
    from eggfotos.ui.panels import WatermarkPanel
    p = WatermarkPanel()
    p.show()
    return p


def _ir_para(p, key):
    p.cmb_perfil.setCurrentIndex(p.cmb_perfil.findData(f"preset:{key}"))


def test_panel_lists_profiles_and_applies_them(painel):
    p = painel
    assert p.cmb_perfil.itemText(0) == "Personalizado"
    assert p.cmb_perfil.count() == 1 + len(presets.PRESETS)
    _ir_para(p, "textura")
    v = p.values()
    t = presets.PRESETS["textura"]
    assert v["metodo"] == "lama" and v["contexto"] == t.contexto and v["grao_forca"] == t.grao_forca
    assert v["expandir"] == t.expandir and v["emenda"] == t.emenda
    assert p.lbl_dica.text() == t.dica
    _ir_para(p, "liso")
    assert p.values()["metodo"] == "telea" and p.values()["raio"] == presets.PRESETS["liso"].raio
    assert p.sl_raio.isVisible() and not p.sl_contexto.isVisible()        # controles seguem o método


def test_touching_a_control_turns_into_custom_and_back(painel):
    p = painel
    _ir_para(p, "textura")
    p.sl_contexto.setValue(p.sl_contexto.value() + 5)
    assert p.cmb_perfil.currentIndex() == 0
    assert p.values()["contexto"] == presets.PRESETS["textura"].contexto + 5   # o ajuste fica
    _ir_para(p, "textura")                                                     # reaplicar restaura
    assert p.values()["contexto"] == presets.PRESETS["textura"].contexto


def test_save_and_delete_user_profile_from_panel(painel, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog, QMessageBox
    p = painel
    _ir_para(p, "grao_alto")
    p.sl_emenda.setValue(33)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Meu ajuste", True))
    p.btn_perfil_salvar.click()
    assert p.cmb_perfil.currentText() == "★ Meu ajuste" and p.btn_perfil_excluir.isEnabled()
    _ir_para(p, "liso")
    p.cmb_perfil.setCurrentIndex(p.cmb_perfil.findData("user:Meu ajuste"))
    assert p.values()["emenda"] == 33 and p.values()["grao_forca"] == presets.PRESETS["grao_alto"].grao_forca
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    p.btn_perfil_excluir.click()
    assert p.cmb_perfil.findData("user:Meu ajuste") == -1 and p.cmb_perfil.currentIndex() == 0
    assert not p.btn_perfil_excluir.isEnabled()
