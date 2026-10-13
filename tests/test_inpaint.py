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
    out = inpaint.remove_with_mask(marked, mask, "lama", expandir=0)
    assert np.all(out[155, 240] == 100)
    assert np.array_equal(out[10, 10], marked[10, 10])


def test_region_op_registered(rgb):
    step = ops.get_operation("remover_marca_regiao").new_step()
    assert ops.apply_step(rgb, step).shape == rgb.shape
    assert "marca" in ops.describe_step(step).lower()
