import numpy as np
import pytest

from eggfotos.core import inpaint
from eggfotos.pipeline import operations as ops


def _scene_with_mark():
    """Gradiente suave com um 'texto' branco por cima."""
    h, w = 200, 300
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([60 + x * 100 // w, 90 + y * 80 // h, np.full_like(x, 140)], axis=2).astype(np.uint8)
    marked = base.copy()
    marked[150:162, 200:280] = 255
    mask = np.zeros((h, w), np.uint8)
    mask[150:162, 200:280] = 255
    return base, marked, mask


@pytest.mark.parametrize("metodo", ["telea", "ns"])
def test_inpaint_removes_mark_keeps_rest(metodo):
    base, marked, mask = _scene_with_mark()
    out = inpaint.remove_with_mask(marked, mask, metodo)
    err_before = np.abs(marked[150:162, 200:280].astype(int) - base[150:162, 200:280]).mean()
    err_after = np.abs(out[150:162, 200:280].astype(int) - base[150:162, 200:280]).mean()
    assert err_after < err_before * 0.15
    far = np.ones(mask.shape, bool)
    far[130:185, 180:300] = False
    assert np.array_equal(out[far], marked[far])  # longe da marca nada muda


def test_empty_mask_is_copy():
    base, marked, _ = _scene_with_mark()
    out = inpaint.remove_with_mask(marked, np.zeros(marked.shape[:2], np.uint8))
    assert np.array_equal(out, marked) and out is not marked


def test_input_untouched_and_alpha_kept(rgba):
    before = rgba.copy()
    mask = np.zeros(rgba.shape[:2], np.uint8)
    mask[40:60, 80:140] = 255
    out = inpaint.remove_with_mask(rgba, mask)
    assert np.array_equal(rgba, before)
    assert out.shape == rgba.shape and np.array_equal(out[..., 3], rgba[..., 3])


def test_grow_mask_and_rect_mask(rgb):
    m = inpaint.rect_mask(rgb.shape, 0.4, 0.4, 0.2, 0.2)
    assert m.sum() > 0
    assert inpaint.grow_mask(m, 100).sum() > inpaint.grow_mask(m, 0).sum()


def test_invalid_method_and_mask_size(rgb):
    with pytest.raises(ValueError):
        inpaint.remove_with_mask(rgb, np.zeros(rgb.shape[:2], np.uint8), "xyz")
    with pytest.raises(ValueError):
        inpaint.remove_with_mask(rgb, np.zeros((5, 5), np.uint8))


def test_lama_without_model_raises(rgb, monkeypatch, tmp_path):
    monkeypatch.setattr(inpaint, "models_dir", lambda: tmp_path)
    monkeypatch.setattr(inpaint, "_lama_session", None)
    mask = inpaint.rect_mask(rgb.shape, 0.3, 0.3, 0.2, 0.2)
    with pytest.raises(inpaint.ModelNotDownloadedError):
        inpaint.remove_with_mask(rgb, mask, "lama")


def test_lama_path_uses_model_output(monkeypatch):
    """Com um LaMa falso (devolve cinza), só a máscara muda e o contexto é preservado."""
    base, marked, mask = _scene_with_mark()
    monkeypatch.setattr(inpaint, "_lama_run", lambda rgb, m: np.full_like(rgb, 100))
    out = inpaint.remove_with_mask(marked, mask, "lama", expandir=0, grao=False)
    assert np.all(out[155, 240] == 100)
    assert np.array_equal(out[10, 10], marked[10, 10])


def test_region_op_registered(rgb):
    step = ops.get_operation("remover_marca_regiao").new_step()
    assert ops.apply_step(rgb, step).shape == rgb.shape
    assert "marca" in ops.describe_step(step).lower()


# ----------------------------------------------------- janelas nativas e grão

def _record_lama(monkeypatch):
    calls = []

    def fake(rgb, m):
        calls.append(rgb.shape[:2])
        return np.full_like(rgb, 90)
    monkeypatch.setattr(inpaint, "_lama_run", fake)
    return calls


def test_lama_uses_native_512_windows_on_big_photo(monkeypatch):
    calls = _record_lama(monkeypatch)
    big = np.full((1600, 2400, 3), 150, np.uint8)
    mask = np.zeros(big.shape[:2], np.uint8)
    mask[1300:1360, 1800:2200] = 255            # marca pequena num canto da foto grande
    out = inpaint.remove_with_mask(big, mask, "lama", expandir=0, grao=False)
    assert calls and all(c == (512, 512) for c in calls)   # nunca reduz a foto
    assert len(calls) <= 2
    assert np.all(out[1330, 2000] == 90)
    assert np.array_equal(out[100, 100], big[100, 100])


def test_lama_tiles_a_large_mark_and_terminates(monkeypatch):
    calls = _record_lama(monkeypatch)
    big = np.full((1600, 2400, 3), 150, np.uint8)
    mask = np.zeros(big.shape[:2], np.uint8)
    mask[400:900, 300:1700] = 255                # marca enorme
    out = inpaint.remove_with_mask(big, mask, "lama", expandir=0, grao=False)
    assert len(calls) <= inpaint.MAX_WINDOWS + 1
    assert np.all(out[600, 1000] == 90) and np.all(out[880, 320] == 90)   # tudo preenchido


def test_small_photo_falls_back_to_downscaled(monkeypatch):
    calls = _record_lama(monkeypatch)
    img = np.full((300, 400, 3), 150, np.uint8)
    mask = inpaint.rect_mask(img.shape, 0.4, 0.4, 0.2, 0.2)
    inpaint.remove_with_mask(img, mask, "lama", grao=False)
    assert len(calls) == 1


def test_restore_grain_matches_surroundings():
    rng = np.random.default_rng(0)
    h, w = 300, 400
    base = np.clip(120 + rng.normal(0, 6, (h, w, 3)), 0, 255).astype(np.uint8)   # foto com grão
    mask = np.zeros((h, w), np.uint8)
    mask[120:180, 150:260] = 255
    smooth = base.copy()
    smooth[mask > 0] = 120                                                       # preenchimento liso
    out = inpaint.restore_grain(base, smooth, mask)
    g = lambda im, reg: inpaint._noise_std(inpaint.cv2.cvtColor(im, inpaint.cv2.COLOR_RGB2GRAY), reg)
    inside = mask > 0
    ring = np.zeros_like(inside)
    ring[100:200, 130:280] = True
    ring &= ~inside
    assert g(smooth, inside) < 0.5
    assert abs(g(out, inside) - g(base, ring)) < 0.25 * g(base, ring)
    assert np.array_equal(out[~inside & ~ring][:5], smooth[~inside & ~ring][:5])   # fora da máscara não muda


def test_grain_is_deterministic():
    base = np.random.default_rng(1).integers(100, 140, (200, 300, 3), dtype=np.uint8)
    mask = inpaint.rect_mask(base.shape, 0.3, 0.3, 0.3, 0.2)
    a = inpaint.remove_with_mask(base, mask, "telea")
    b = inpaint.remove_with_mask(base, mask, "telea")
    assert np.array_equal(a, b)


# ------------------------------------------------ ajuste aos traços (snap)

def _textured_with_text(cor, alpha=0.4, seed=2):
    rng = np.random.default_rng(seed)
    h, w = 400, 600
    y, x = np.mgrid[0:h, 0:w]
    base = (110 + 50 * np.sin(x / 70.0) + 35 * np.cos(y / 55.0))[..., None] * np.array([1.0, 0.95, 0.85])
    base = np.clip(base + rng.normal(0, 4, (h, w, 1)), 0, 255).astype(np.float32)
    txt = np.zeros((h, w), np.uint8)
    inpaint.cv2.putText(txt, "COPYRIGHT", (120, 230), inpaint.cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, 5)
    a = (txt.astype(np.float32) / 255.0 * alpha)[..., None]
    marked = np.clip(base * (1 - a) + cor * a + 0.5, 0, 255).astype(np.uint8)
    loose = np.zeros((h, w), np.uint8)
    loose[170:260, 100:520] = 255                  # retângulo solto em volta do texto
    return np.clip(base, 0, 255).astype(np.uint8), marked, txt > 0, loose


@pytest.mark.parametrize("cor", [255.0, 0.0])
def test_snap_tightens_mask_onto_the_text(cor):
    clean, marked, truth, loose = _textured_with_text(cor)
    snapped = inpaint.snap_to_strokes(marked, loose) > 0
    assert snapped.sum() < 0.4 * (loose > 0).sum()                 # bem mais justa que o retângulo
    assert (snapped & truth).sum() / truth.sum() > 0.8            # pega a maior parte do texto
    assert (snapped & ~(inpaint.cv2.dilate(truth.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)).sum() \
        < 0.35 * snapped.sum()                                     # pouco falso positivo


def test_snap_improves_the_inpaint_result():
    clean, marked, truth, loose = _textured_with_text(255.0)
    region = loose > 0
    err = lambda out: np.abs(out.astype(int) - clean)[region].mean()
    out_loose = inpaint.remove_with_mask(marked, loose, "telea", expandir=0)
    snapped = inpaint.snap_to_strokes(marked, loose)
    out_snap = inpaint.remove_with_mask(marked, snapped, "telea", expandir=30)
    assert err(out_snap) < err(out_loose)


def test_snap_polarity_and_validation():
    clean, marked, truth, loose = _textured_with_text(255.0)
    cov = lambda pol: (inpaint.snap_to_strokes(marked, loose, pol, 40) > 0)[truth].mean()
    assert cov("clara") > 0.8 and cov("escura") < 0.2          # marca branca: só "clara" acerta
    assert cov("auto") > 0.8
    assert inpaint.snap_to_strokes(marked, np.zeros_like(loose)).sum() == 0
    with pytest.raises(ValueError):
        inpaint.snap_to_strokes(marked, loose, "verde")
    low = inpaint.snap_to_strokes(marked, loose, sensibilidade=0).sum()
    high = inpaint.snap_to_strokes(marked, loose, sensibilidade=100).sum()
    assert high >= low


# ------------------------------------------- controles do preenchimento (LaMa etc.)

def test_grain_strength_scales_the_noise():
    rng = np.random.default_rng(1)
    base = np.clip(120 + rng.normal(0, 6, (300, 400, 3)), 0, 255).astype(np.uint8)
    mask = inpaint.rect_mask(base.shape, 0.3, 0.3, 0.4, 0.3)
    region = inpaint.grow_mask(mask, 0) > 0
    gray = lambda im: inpaint._noise_std(inpaint.cv2.cvtColor(im, inpaint.cv2.COLOR_RGB2GRAY), region)
    none = inpaint.remove_with_mask(base, mask, "telea", grao_forca=0)
    normal = inpaint.remove_with_mask(base, mask, "telea", grao_forca=100)
    double = inpaint.remove_with_mask(base, mask, "telea", grao_forca=200)
    assert gray(none) < gray(normal) < gray(double)
    assert np.array_equal(none, inpaint.remove_with_mask(base, mask, "telea", grao=False))


def test_seam_smoothing_controls_the_edge():
    base = np.random.default_rng(2).integers(80, 120, (300, 400, 3), dtype=np.uint8)
    mask = inpaint.rect_mask(base.shape, 0.4, 0.4, 0.2, 0.2)
    filled = np.full_like(base, 250)
    hard = inpaint._composite(base, filled, mask, emenda=0)
    soft = inpaint._composite(base, filled, mask, emenda=100)
    y, x = 120, 158                                    # logo fora da borda esquerda da máscara
    assert np.array_equal(hard[y, x], base[y, x])      # emenda 0: corte seco, nada vaza para fora
    assert soft[y, x].mean() > base[y, x].mean() + 1   # emenda 100: a borda se mistura para fora
    assert np.all(soft[150, 200] == 250) and np.all(hard[150, 200] == 250)   # dentro: sempre cobre a marca


def test_lama_context_changes_the_window_layout(monkeypatch):
    big = np.full((1600, 2400, 3), 150, np.uint8)
    mask = np.zeros(big.shape[:2], np.uint8)
    mask[400:560, 300:1700] = 255
    n = {}
    for ctx in (0, 100):
        calls = []
        monkeypatch.setattr(inpaint, "_lama_run", lambda rgb, m, c=calls: (c.append(1), np.full_like(rgb, 90))[1])
        inpaint.remove_with_mask(big, mask, "lama", expandir=0, grao=False, contexto=ctx)
        n[ctx] = len(calls)
    assert n[100] > n[0]            # mais contexto = menos área útil por janela = mais janelas


def test_new_params_in_operations(rgb, marks_tmp):
    for key in ("remover_marca_regiao", "remover_marca_modelo"):
        names = {p.name for p in ops.get_operation(key).params}
        assert {"contexto", "emenda", "grao_forca", "grao"} <= names
    old = ops.normalize_step({"op": "remover_marca_regiao", "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2, "grao": False})
    assert old["grao"] is False and old["grao_forca"] == 100 and old["contexto"] == 40   # receita antiga continua válida
