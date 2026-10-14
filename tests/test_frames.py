import numpy as np
import pytest

from eggfotos.core import frames
from eggfotos.pipeline import operations as ops

KEYS = frames.frame_keys()


@pytest.fixture
def foto():
    h, w = 240, 360
    y, x = np.mgrid[0:h, 0:w]
    return np.stack([x * 255 // w, y * 255 // h, np.full_like(x, 120)], 2).astype(np.uint8)


def test_registry_has_the_expected_frames():
    assert len(KEYS) >= 15
    for k in ("polaroid", "rolo35", "negativo35", "slide", "super8", "vhs", "crt", "cinema", "galeria"):
        assert k in KEYS
    groups = {frames.FRAMES[k].group for k in KEYS}
    assert groups == {"Papel e galeria", "Cinema e filme", "Eletrônicos"}


@pytest.mark.parametrize("key", KEYS)
def test_every_frame_returns_rgb_and_keeps_the_input(foto, key):
    antes = foto.copy()
    out = frames.apply_frame(foto, key)
    assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3
    assert out.shape[0] >= foto.shape[0] and out.shape[1] >= foto.shape[1]   # moldura nunca encolhe
    assert np.array_equal(foto, antes)                                       # entrada intacta
    assert not np.array_equal(out[:foto.shape[0], :foto.shape[1]], foto) or out.shape != foto.shape


@pytest.mark.parametrize("key", KEYS)
def test_frames_are_deterministic(foto, key):
    assert np.array_equal(frames.apply_frame(foto, key, texto="X"), frames.apply_frame(foto, key, texto="X"))


@pytest.mark.parametrize("key", KEYS)
def test_rgba_input_is_flattened(rgba, key):
    out = frames.apply_frame(rgba, key)
    assert out.shape[2] == 3


@pytest.mark.parametrize("key", [k for k in KEYS if k not in ("vhs", "crt", "super8", "rolo35", "negativo35", "slide", "cartao", "fita", "album")])
def test_border_frames_keep_the_photo_pixels_intact(foto, key):
    out = frames.apply_frame(foto, key, espessura=50)
    # a foto original aparece inteira, sem alteração, em algum lugar da moldura
    res = frames.cv2.matchTemplate(out, foto, frames.cv2.TM_SQDIFF)      # acha a posição (valor ≈ 0)
    y0, x0 = np.unravel_index(res.argmin(), res.shape)
    h, w = foto.shape[:2]
    assert np.array_equal(out[y0:y0 + h, x0:x0 + w], foto), key           # e confere pixel a pixel


def test_thickness_scales_with_the_control_and_the_photo(foto):
    fino = frames.apply_frame(foto, "simples", espessura=20)
    padrao = frames.apply_frame(foto, "simples", espessura=50)
    grosso = frames.apply_frame(foto, "simples", espessura=100)
    assert fino.shape[0] < padrao.shape[0] < grosso.shape[0]
    assert frames.apply_frame(foto, "simples", espessura=0).shape == foto.shape       # 0 = sem borda
    big = np.kron(foto, np.ones((3, 3, 1), np.uint8))                                  # foto 3× maior
    r_small = (padrao.shape[0] - foto.shape[0]) / foto.shape[0]
    r_big = (frames.apply_frame(big, "simples").shape[0] - big.shape[0]) / big.shape[0]
    assert abs(r_small - r_big) < 0.01                                                 # proporção igual (regra 3)


def test_color_and_text_controls(foto):
    azul = frames.apply_frame(foto, "simples", cor="#0000ff", cor_padrao=False)
    branco = frames.apply_frame(foto, "simples", cor="#0000ff", cor_padrao=True)
    assert tuple(azul[0, 0]) == (0, 0, 255) and tuple(branco[0, 0]) == (255, 255, 255)
    com = frames.apply_frame(foto, "polaroid", texto="Praia 2026")
    sem = frames.apply_frame(foto, "polaroid")
    assert not np.array_equal(com, sem) and com.shape == sem.shape
    assert not np.array_equal(frames.apply_frame(foto, "vhs", texto="MAR 03 1998"), frames.apply_frame(foto, "vhs", texto="JAN 01 2000"))


def test_film_frames_look_like_film(foto):
    rolo = frames.apply_frame(foto, "rolo35")
    assert rolo.shape[1] > 1.4 * foto.shape[1] and rolo.shape[0] > foto.shape[0]         # tira larga com furos
    assert rolo[:3, :3].mean() < 40                                                         # base escura
    neg = frames.apply_frame(foto, "negativo35")
    h = neg.shape[0]
    centro = neg[h // 2, neg.shape[1] // 2].astype(int)
    assert centro[0] > centro[2]                                                            # negativo laranja: R > B


def test_overlays_keep_the_size(foto):
    assert frames.apply_frame(foto, "vhs").shape == foto.shape


def test_unknown_frame(foto):
    with pytest.raises(ValueError):
        frames.apply_frame(foto, "nao_existe")


def test_operation_registered_and_described(foto):
    step = ops.get_operation("moldura").new_step(nome="slide", texto="7A")
    out = ops.apply_step(foto, step)
    assert out.shape[0] > foto.shape[0]
    assert "Slide" in ops.describe_step(step)
    old = ops.normalize_step({"op": "moldura", "nome": "vhs"})
    assert old["espessura"] == 50 and old["cor_padrao"] is True and old["texto"] == ""


def test_large_photo_is_fast_enough():
    import time
    big = np.random.default_rng(0).integers(0, 255, (3000, 4500, 3), dtype=np.uint8)
    for k in ("polaroid", "rolo35", "vhs", "cartao"):
        t = time.time()
        frames.apply_frame(big, k)
        assert time.time() - t < 12, k
