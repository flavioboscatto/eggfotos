"""Varinha mágica dos Ajustes: sugestão automática sem IA."""
import os
import time

import cv2
import numpy as np
import pytest

from eggfotos.core import adjustments, autoadjust
from eggfotos.core.imgutils import luminance, to_float

H, W = 360, 540


def _foto(seed=3, sat=0.55):
    """Cena sintética com detalhe em várias escalas, tons de 3% a 95% e cor moderada."""
    rng = np.random.default_rng(seed)
    ch = []
    for _ in range(3):
        acc = np.zeros((H, W), np.float32)
        for sz, amp in ((6, 1.0), (14, 0.7), (40, 0.5), (120, 0.4)):
            n = rng.standard_normal((H // sz + 2, W // sz + 2)).astype(np.float32)
            acc += amp * cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
        ch.append(acc)
    base = ch[0] + 0.0 * ch[1]
    lo, hi = np.percentile(base, (1, 99))
    lum = np.clip((base - lo) / (hi - lo), 0, 1) * 0.92 + 0.03
    tint = np.stack([ch[0] * 0.0 + (ch[1] - ch[1].mean()) * 0.06 * sat * 2,
                     (ch[2] - ch[2].mean()) * 0.06 * sat * 2,
                     -(ch[1] - ch[1].mean()) * 0.05 * sat * 2], 2)
    img = np.clip(lum[..., None] + tint, 0, 1)
    return (img * 255 + 0.5).astype(np.uint8)


def _f(a):
    return a.astype(np.float32)


def test_foto_equilibrada_quase_nao_muda():
    s = autoadjust.suggest_adjustments(_foto())
    assert abs(s.values["exposicao"]) <= 15 and s.values["contraste"] == 0
    assert s.values["realces"] == 0 and s.values["ruido"] == 0 and not s.values["temperatura"]


def test_foto_escura_sobe_exposicao_e_melhora():
    img = np.clip(_f(_foto()) * 0.4, 0, 255).astype(np.uint8)
    s = autoadjust.suggest_adjustments(img)
    assert s.values["exposicao"] > 15
    out = adjustments.adjust(img, **s.values)
    alvo = autoadjust.TARGET_MEDIAN
    assert abs(np.median(luminance(to_float(out))) - alvo) < abs(np.median(luminance(to_float(img))) - alvo)
    assert any("escura" in m for m in s.motivos)


def test_foto_clara_demais_reduz_exposicao():
    img = np.clip(_f(_foto()) * 0.5 + 150, 0, 255).astype(np.uint8)
    assert autoadjust.suggest_adjustments(img).values["exposicao"] < -10


def test_foto_lavada_ganha_contraste():
    img = np.clip((_f(_foto()) - 128) * 0.45 + 128, 0, 255).astype(np.uint8)
    s = autoadjust.suggest_adjustments(img)
    assert s.values["contraste"] >= 15
    out = adjustments.adjust(img, **s.values)
    assert np.ptp(out) > np.ptp(img) or out.std() > img.std()


def test_realces_estourados_sao_atenuados():
    img = np.clip(_f(_foto()) * 1.5 + 30, 0, 255).astype(np.uint8)
    s = autoadjust.suggest_adjustments(img)
    assert s.values["realces"] < 0 and s.values["realces"] >= -40


def test_sombras_levantadas_quando_ha_muito_escuro():
    img = _foto()
    g = to_float(img)
    img = (np.where(np.arange(W)[None, :, None] < W * 0.4, g * 0.12, g) * 255).astype(np.uint8)
    assert autoadjust.suggest_adjustments(img).values["sombras"] > 0


def test_dominante_azulada_em_tons_neutros_esquenta():
    img = _foto(sat=0.0)
    img = np.clip(_f(img) * np.array([0.9, 0.97, 1.1]), 0, 255).astype(np.uint8)
    s = autoadjust.suggest_adjustments(img)
    assert s.values["temperatura"] > 0
    out = adjustments.adjust(img, **s.values)
    assert abs(np.log(out[..., 0].mean() / out[..., 2].mean())) < abs(np.log(img[..., 0].mean() / img[..., 2].mean()))


def test_dominante_amarelada_esfria():
    img = np.clip(_f(_foto(sat=0.0)) * np.array([1.1, 1.0, 0.9]), 0, 255).astype(np.uint8)
    assert autoadjust.suggest_adjustments(img).values["temperatura"] < 0


def test_foto_mole_ganha_nitidez_e_foto_nitida_nao():
    mole = cv2.GaussianBlur(_foto(), (0, 0), 3)
    assert autoadjust.suggest_adjustments(mole).values["nitidez"] >= 28
    assert autoadjust.suggest_adjustments(_foto()).values["nitidez"] <= 15


def test_ruido_ativa_remocao_e_nao_afia():
    base = _foto()
    sujo = np.clip(_f(base) + np.random.default_rng(1).standard_normal(base.shape) * 14, 0, 255).astype(np.uint8)
    s = autoadjust.suggest_adjustments(sujo)
    assert s.values["ruido"] >= 15 and s.values["nitidez"] <= 15
    assert autoadjust.suggest_adjustments(base).values["ruido"] == 0
    assert autoadjust.estimate_noise(sujo) > 3 * max(autoadjust.estimate_noise(base), 0.5)


def test_cores_apagadas_ganham_saturacao():
    img = _foto(sat=1.0)
    g = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    fraco = np.clip(_f(g) * 0.8 + _f(img) * 0.2, 0, 255).astype(np.uint8)
    s = autoadjust.suggest_adjustments(fraco)
    assert s.values["saturacao"] >= 0


def test_valores_dentro_das_faixas_e_chaves():
    for img in (_foto(), np.zeros((H, W, 3), np.uint8), np.full((H, W, 3), 255, np.uint8)):
        s = autoadjust.suggest_adjustments(img)
        assert set(s.values) == set(adjustments.ADJUSTMENT_KEYS)
        for k, v in s.values.items():
            lo, hi = adjustments.adjustment_range(k)
            assert lo <= v <= hi
        adjustments.adjust(img, **s.values)          # o ajuste aceita o que foi sugerido


def test_intensidade_escala_a_sugestao():
    s = autoadjust.suggest_adjustments(np.clip(_f(_foto()) * 0.4, 0, 255).astype(np.uint8))
    meio, dobro, zero = s.scaled(50), s.scaled(150), s.scaled(0)
    assert abs(meio["exposicao"]) < abs(s.values["exposicao"]) < abs(dobro["exposicao"]) or dobro["exposicao"] == 60
    assert not any(zero.values())


def test_deterministico_nao_altera_entrada_e_aceita_rgba():
    img = np.clip(_f(_foto()) * 0.4, 0, 255).astype(np.uint8)
    antes = img.copy()
    a, b = autoadjust.suggest_adjustments(img), autoadjust.suggest_adjustments(img)
    assert np.array_equal(img, antes) and a.values == b.values
    rgba = np.dstack([img, np.full((H, W), 255, np.uint8)])
    assert autoadjust.suggest_adjustments(rgba).values == a.values


def test_foto_grande_e_rapida():
    big = cv2.resize(_foto(), (4500, 3000), interpolation=cv2.INTER_LINEAR)
    t = time.time()
    autoadjust.suggest_adjustments(big)
    assert time.time() - t < 3.0


def test_resolucao_nao_muda_a_sugestao_de_luz():
    img = np.clip(_f(_foto()) * 0.4, 0, 255).astype(np.uint8)
    a = autoadjust.suggest_adjustments(img).values
    b = autoadjust.suggest_adjustments(cv2.resize(img, (W * 2, H * 2), interpolation=cv2.INTER_LINEAR)).values
    assert abs(a["exposicao"] - b["exposicao"]) <= 3 and abs(a["contraste"] - b["contraste"]) <= 5


# ----------------------------------------------------------------- interface

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def janela(qapp):
    from eggfotos.ui.main_window import MainWindow
    w = MainWindow()
    img = np.clip(_f(_foto()) * 0.4, 0, 255).astype(np.uint8)
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


def test_botao_preenche_sliders_motivos_e_aplica(janela):
    w = janela
    p = w.adjust_panel
    assert p.box_auto.isHidden()
    p.btn_auto.click(); w.esperar()
    assert not p.box_auto.isHidden() and p.values()["exposicao"] > 15 and "escura" in p.lbl_auto.text()
    antes, expo = w.image.copy(), p.values()["exposicao"]
    p.btn_apply.click(); w.esperar()
    step = w.history._entries[-1].step
    assert step["op"] == "ajustes" and step["exposicao"] == expo
    assert w.image.mean() > antes.mean()


def test_intensidade_reescala_os_sliders(janela):
    w = janela
    p = w.adjust_panel
    p.btn_auto.click(); w.esperar()
    cheio = p.values()["exposicao"]
    p.sl_auto.setValue(50)
    assert 0 < p.values()["exposicao"] < cheio
    p.sl_auto.setValue(0)
    assert not any(p.values().values())


def test_zerar_esconde_a_sugestao(janela):
    w = janela
    p = w.adjust_panel
    p.btn_auto.click(); w.esperar()
    p.btn_reset.click()
    assert p.box_auto.isHidden() and not any(p.values().values())
