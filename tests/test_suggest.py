"""Pré-análise sem IA: sugestão de ajustes e procura de marca d'água."""
import cv2
import numpy as np
import pytest

from eggfotos.core import suggest

H, W = 600, 900


def _scene(kind, seed=1, grain=1.0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W]
    if kind == "sky":
        base = np.stack([100 + 60 * y / H, 150 + 50 * y / H, 230 - 30 * y / H], 2)
    else:
        n = cv2.resize(rng.standard_normal((H // 6 + 1, W // 6 + 1)).astype(np.float32), (W, H),
                       interpolation=cv2.INTER_CUBIC)
        base = np.stack([110 + 25 * n, 140 + 25 * n, 60 + 20 * n], 2)
    return np.clip(base + rng.standard_normal((H, W, 3)) * grain, 0, 255).astype(np.uint8)


def _mark(img, alpha=0.45, color=255):
    ov = np.zeros(img.shape[:2], np.uint8)
    cv2.putText(ov, "WWW.EXEMPLO.COM", (560, 540), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
    a = (ov / 255.0 * alpha)[..., None]
    return (img * (1 - a) + color * a).astype(np.uint8), ov


def _loose_mask():
    m = np.zeros((H, W), np.uint8)
    m[490:560, 540:880] = 255
    return m


def test_fundo_liso_sugere_preenchimento_rapido():
    wm, _ = _mark(_scene("sky"))
    s = suggest.suggest_settings(wm, _loose_mask())
    assert s.values["metodo"] == "telea"
    assert s.perfil == "liso"
    assert s.polaridade == "clara"
    assert s.apertar                      # marcação solta (retângulo cheio)
    assert any("liso" in m for m in s.motivos)


def test_fundo_texturado_sugere_lama_com_mais_contexto():
    wm, _ = _mark(_scene("tex"))
    s = suggest.suggest_settings(wm, _loose_mask())
    assert s.values["metodo"] == "lama"
    assert s.values["contexto"] >= 50
    assert s.medidas["nivel"] == "alta"


def test_grao_alto_e_baixo():
    lim, _ = _mark(_scene("sky", grain=0.3))
    sujo, _ = _mark(_scene("sky", grain=7.0))
    assert suggest.suggest_settings(lim, _loose_mask()).values["grao_forca"] < 100
    assert suggest.suggest_settings(sujo, _loose_mask()).values["grao_forca"] > 100


def test_marca_grande_usa_lama_e_pouco_contexto():
    wm, _ = _mark(_scene("sky"))
    m = np.zeros((H, W), np.uint8)
    m[100:500, 100:850] = 255
    s = suggest.suggest_settings(wm, m)
    assert s.values["metodo"] == "lama"
    assert s.values["contexto"] <= 40
    assert s.perfil == "grande"


def test_valores_dentro_das_faixas_e_sem_alterar_entrada():
    from eggfotos.core.presets import RANGES
    wm, _ = _mark(_scene("tex"))
    antes = wm.copy()
    s = suggest.suggest_settings(wm, _loose_mask())
    assert np.array_equal(wm, antes)
    for k, (lo, hi) in RANGES.items():
        assert lo <= s.values[k] <= hi


def test_independe_da_resolucao():
    wm, _ = _mark(_scene("sky"))
    a = suggest.suggest_settings(wm, _loose_mask())
    big = cv2.resize(wm, (W * 2, H * 2), interpolation=cv2.INTER_LINEAR)
    mbig = cv2.resize(_loose_mask(), (W * 2, H * 2), interpolation=cv2.INTER_NEAREST)
    b = suggest.suggest_settings(big, mbig)
    assert a.values["metodo"] == b.values["metodo"]
    assert a.perfil == b.perfil


def test_sem_marcacao_ou_tamanho_errado():
    with pytest.raises(ValueError):
        suggest.suggest_settings(_scene("sky"), np.zeros((H, W), np.uint8))
    with pytest.raises(ValueError):
        suggest.suggest_settings(_scene("sky"), np.zeros((10, 10), np.uint8))


def test_rgba_aceito():
    wm, _ = _mark(_scene("sky"))
    rgba = np.dstack([wm, np.full((H, W), 255, np.uint8)])
    assert suggest.suggest_settings(rgba, _loose_mask()).values["metodo"] == "telea"


# -------------------------------------------------------------- procurar marca

def _iou(c, box, shape=(H, W)):
    x0, y0 = max(c.x, box[0]), max(c.y, box[1])
    x1, y1 = min(c.x + c.w, box[0] + box[2]), min(c.y + c.h, box[1] + box[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / (c.w * c.h + box[2] * box[3] - inter)


@pytest.mark.parametrize("cor,pol", [(255, "clara"), (0, "escura")])
def test_acha_texto_em_fundo_liso(cor, pol):
    wm, ov = _mark(_scene("sky"), alpha=0.5, color=cor)
    res = suggest.find_candidates(wm)
    assert res.itens
    ys, xs = np.where(ov > 0)
    verdade = (xs.min() / W, ys.min() / H, (xs.max() - xs.min()) / W, (ys.max() - ys.min()) / H)
    melhor = res.itens[0]
    assert _iou(melhor, verdade) > 0.4
    assert melhor.polaridade == pol
    assert "inferior" in melhor.lugar and "direito" in melhor.lugar
    assert 0 <= melhor.x and melhor.x + melhor.w <= 1 and melhor.y + melhor.h <= 1


def test_foto_lisa_sem_marca_nao_acha_nada():
    assert suggest.find_candidates(_scene("sky")).itens == []


def test_foto_muito_texturada_nao_inventa_candidatas():
    wm, _ = _mark(_scene("tex"))
    assert suggest.find_candidates(wm).itens == []


def test_procura_e_deterministica_e_nao_altera_entrada():
    wm, _ = _mark(_scene("sky"))
    antes = wm.copy()
    a, b = suggest.find_candidates(wm), suggest.find_candidates(wm)
    assert np.array_equal(wm, antes)
    assert [c.rect() for c in a.itens] == [c.rect() for c in b.itens]


def test_marca_em_mosaico_e_detectada_como_repetida():
    img = _scene("sky")
    ov = np.zeros((H, W), np.uint8)
    for yy in range(80, H, 150):
        for xx in range(20, W, 280):
            cv2.putText(ov, "LOGO", (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2, cv2.LINE_AA)
    a = (ov / 255.0 * 0.4)[..., None]
    wm = (img * (1 - a) + 255 * a).astype(np.uint8)
    assert suggest.find_candidates(wm).repetida
