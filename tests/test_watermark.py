import cv2
import numpy as np
import pytest

from eggfotos.core import watermark as wm
from eggfotos.pipeline import operations as ops
from eggfotos.pipeline.recipe import Recipe


# ---------------------------------------------------------- máscara ancorada

def _corner_mask(h, w, wf=0.25, hf=0.06, margin=0.03):
    """Marca no canto inferior direito, com tamanho/margem proporcionais à largura."""
    m = np.zeros((h, w), np.uint8)
    bw, bh, mg = round(wf * w), round(hf * w), round(margin * w)
    m[h - mg - bh: h - mg, w - mg - bw: w - mg] = 255
    return m


def test_mask_template_scales_to_other_sizes():
    t = wm.make_mask_template(_corner_mask(600, 800))
    assert (t.ax, t.ay) == ("r", "b")
    for h, w in ((900, 1200), (300, 400), (500, 1000)):   # inclusive outra proporção
        placed = wm.place_mask(t, (h, w))
        expected = _corner_mask(h, w)
        inter, union = ((placed > 0) & (expected > 0)).sum(), ((placed > 0) | (expected > 0)).sum()
        assert inter / union > 0.93, (h, w)


def test_mask_template_anchors():
    for (cy, cx), anchors in {(0.1, 0.1): ("l", "t"), (0.5, 0.5): ("c", "c"), (0.9, 0.1): ("l", "b")}.items():
        m = np.zeros((300, 600), np.uint8)
        m[int(cy * 300) - 5:int(cy * 300) + 5, int(cx * 600) - 20:int(cx * 600) + 20] = 255
        t = wm.make_mask_template(m)
        assert (t.ax, t.ay) == anchors
        assert wm.place_mask(t, m.shape).sum() > 0


def test_empty_mask_rejected():
    with pytest.raises(ValueError):
        wm.make_mask_template(np.zeros((10, 10), np.uint8))


def test_remove_with_template_roundtrip(marks_tmp):
    h, w = 300, 400
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([60 + x * 100 // w, 90 + y * 80 // h, np.full_like(x, 140)], 2).astype(np.uint8)
    mask = _corner_mask(h, w)
    marked = base.copy()
    marked[mask > 0] = 255
    wm.save_mask_template("logo", wm.make_mask_template(mask))
    assert wm.list_templates("mask") == ["logo"]
    out = wm.remove_with_template(marked, "logo")
    assert np.abs(out.astype(int) - base).mean() < 0.15 * np.abs(marked.astype(int) - base).mean() + 1


def test_missing_template_error(marks_tmp):
    with pytest.raises(FileNotFoundError):
        wm.remove_with_template(np.zeros((20, 20, 3), np.uint8), "nao_existe")


def test_recipe_validate_flags_missing_template(marks_tmp):
    r = Recipe("x", [{"op": "remover_marca_modelo", "nome": "nao_existe"}])
    errs = r.validate()
    assert errs and "nao_existe" in errs[0]
    wm.save_mask_template("ok", wm.make_mask_template(_corner_mask(100, 100)))
    assert not Recipe("x", [{"op": "remover_marca_modelo", "nome": "ok"}]).validate()


# ------------------------------------------------------------ alpha estimado

def _photos(n, h=160, w=240, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        lo = cv2.resize(rng.integers(0, 256, (6, 9, 3), dtype=np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
        out.append(np.clip(lo.astype(np.float32) + rng.normal(0, 6, lo.shape), 0, 255).astype(np.uint8))
    return out


def _watermark_alpha(h=160, w=240, a=0.35):
    m = np.zeros((h, w), np.uint8)
    cv2.putText(m, "LOGO", (60, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 255, 4)
    return cv2.GaussianBlur(m, (0, 0), 0.8).astype(np.float32) / 255.0 * a


def _apply_wm(img, alpha, cor):
    a = alpha[..., None]
    return np.clip(img * (1 - a) + cor * a + 0.5, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("cor", [255, 0])
def test_estimate_and_remove_translucent_mark(cor):
    alpha = _watermark_alpha()
    train = [_apply_wm(p.astype(np.float32), alpha, cor) for p in _photos(40)]
    t = wm.estimate_alpha(train, cor)
    test_clean = _photos(1, seed=99)[0]
    test_marked = _apply_wm(test_clean.astype(np.float32), alpha, cor)
    out = wm.remove_alpha(test_marked, t)
    region = alpha > 0.05
    before = np.abs(test_marked.astype(int) - test_clean)[region].mean()
    after = np.abs(out.astype(int) - test_clean)[region].mean()
    assert after < before * 0.35, (before, after)
    assert np.abs(out.astype(int) - test_clean)[~region].mean() < 1.5   # fora da marca quase nada muda


def test_too_few_examples():
    with pytest.raises(ValueError):
        wm.estimate_alpha(_photos(3))


def test_alpha_template_save_load_and_op(marks_tmp):
    alpha = _watermark_alpha()
    t = wm.estimate_alpha([_apply_wm(p.astype(np.float32), alpha, 255) for p in _photos(30)], 255)
    wm.save_alpha_template("agua", t)
    t2 = wm.load_alpha_template("agua")
    assert t2.cor == 255 and t2.n == 30 and np.abs(t2.alpha - t.alpha).max() < 1e-4
    img = _apply_wm(_photos(1, seed=5)[0].astype(np.float32), alpha, 255)
    out = ops.apply_step(img, {"op": "remover_marca_estimada", "nome": "agua", "forca": 100})
    assert out.shape == img.shape
    assert "agua" in ops.describe_step({"op": "remover_marca_estimada", "nome": "agua", "forca": 100})


def test_remove_alpha_rejects_other_aspect():
    t = wm.AlphaTemplate(np.full((100, 200), 0.2, np.float32), 255, 10)
    with pytest.raises(ValueError):
        wm.remove_alpha(np.zeros((200, 100, 3), np.uint8), t)


def test_learn_from_folder(tmp_path, marks_tmp):
    from eggfotos.core.image_io import save_image
    alpha = _watermark_alpha()
    folder = tmp_path / "fotos"
    folder.mkdir()
    for i, p in enumerate(_photos(24)):
        save_image(_apply_wm(p.astype(np.float32), alpha, 255), folder / f"f{i}.png")
    t, skipped = wm.learn_from_folder(folder, "pasta", 255)
    assert t.n == 24 and skipped == 0 and wm.template_exists("pasta", "alpha")
