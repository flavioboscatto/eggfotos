import numpy as np
import pytest

from eggfotos.core import background as bg
from eggfotos.pipeline import operations as ops


def test_without_model_raises_friendly_error(rgb, monkeypatch, tmp_path):
    monkeypatch.setattr(bg, "models_dir", lambda: tmp_path)
    bg._sessions.clear()
    assert not bg.model_available("rapido")
    with pytest.raises(bg.ModelNotDownloadedError):
        bg.compute_matte(rgb, "rapido")


def test_unknown_model(rgb):
    with pytest.raises(ValueError):
        bg.compute_matte(rgb, "xyz")


def test_transparent_background(rgb, fake_matte):
    out = bg.remove_background(rgb, suavizar=0)
    assert out.shape == rgb.shape[:2] + (4,)
    assert out[0, 0, 3] == 0 and out[60, 100, 3] == 255
    assert np.array_equal(out[60, 100, :3], rgb[60, 100])


def test_solid_color_background(rgb, fake_matte):
    out = bg.remove_background(rgb, suavizar=0, fundo="cor", cor="#00ff00")
    assert out.shape == rgb.shape
    assert tuple(out[0, 0]) == (0, 255, 0)
    assert np.array_equal(out[60, 100], rgb[60, 100])


def test_bad_fundo(rgb, fake_matte):
    with pytest.raises(ValueError):
        bg.remove_background(rgb, fundo="azul")


def test_input_not_modified_and_existing_alpha_kept(rgba, fake_matte):
    before = rgba.copy()
    out = bg.remove_background(rgba, suavizar=0)
    assert np.array_equal(rgba, before)
    assert out[60, 10, 3] == 0  # alpha original (coluna transparente) é respeitado


def test_refine_alpha_erode_dilate_and_smooth():
    a = np.zeros((200, 200), np.uint8)
    a[50:150, 50:150] = 255
    assert bg.refine_alpha(a, 100).sum() < a.sum()
    assert bg.refine_alpha(a, -100).sum() > a.sum()
    soft = bg.refine_alpha(a, 0, 100)
    assert ((soft > 0) & (soft < 255)).any()
    assert np.array_equal(bg.refine_alpha(a, 0, 0), a)


def test_decontaminate_removes_background_halo():
    rgb = np.zeros((100, 100, 3), np.uint8)
    rgb[...] = (0, 255, 0)           # fundo verde
    rgb[30:70, 30:70] = (200, 0, 0)  # objeto vermelho
    rgb[28:30, 30:70] = (100, 128, 0)  # halo misturado na borda
    alpha = np.zeros((100, 100), np.uint8)
    alpha[30:70, 30:70] = 255
    alpha[28:30, 30:70] = 128
    out = bg.decontaminate(rgb, alpha)
    assert out[29, 50, 1] < rgb[29, 50, 1]       # menos verde no halo
    assert np.array_equal(out[50, 50], rgb[50, 50])  # interior intacto


def test_operation_registered_and_described(rgb, fake_matte):
    step = ops.get_operation("remover_fundo").new_step(fundo="cor", cor="#112233")
    out = ops.apply_step(rgb, step)
    assert out.shape == rgb.shape
    assert "Remover fundo" in ops.describe_step(step)
