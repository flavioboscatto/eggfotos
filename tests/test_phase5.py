import numpy as np
import pytest

from eggfotos.core import background as bg
from eggfotos.core import faces
from eggfotos.pipeline import operations as ops


def _noisy(h=120, w=200):
    return np.random.default_rng(1).integers(0, 255, (h, w, 3), dtype=np.uint8)


# ----------------------------------------------------------- desfoque de fundo

def test_blur_background_keeps_subject_blurs_rest(fake_matte):
    img = _noisy()
    out = bg.blur_background_auto(img, intensidade=60, suavizar=0)
    assert out.shape == img.shape
    assert np.array_equal(out[60, 100], img[60, 100])        # centro = objeto, intacto
    assert np.abs(out[:20].astype(int) - img[:20]).mean() > 20  # periferia desfocada


def test_blur_background_preserves_alpha(rgba, fake_matte):
    out = bg.blur_background_auto(rgba)
    assert out.shape == rgba.shape
    assert np.array_equal(out[..., 3], rgba[..., 3])


# --------------------------------------------------------------------- rostos

def test_sensitivity_threshold_monotonic():
    assert faces.score_threshold(100) < faces.score_threshold(60) < faces.score_threshold(0)


def test_blur_boxes_filters_by_sensitivity(fake_faces):
    img = _noisy()
    low = faces.blur_boxes(img, fake_faces, sensibilidade=10, tipo="solido")   # só o rosto forte
    high = faces.blur_boxes(img, fake_faces, sensibilidade=100, tipo="solido")  # os dois
    weak = (int(0.5 * 120 + 0.25 * 120), int(0.7 * 200))  # centro do rosto fraco (y, x)
    assert np.array_equal(low[weak], img[weak])
    assert not np.array_equal(high[weak], img[weak])
    strong = (int(0.25 * 120), int(0.2 * 200))
    assert tuple(low[strong]) == (0, 0, 0)


def test_blur_boxes_no_faces_is_copy():
    img = _noisy()
    out = faces.blur_boxes(img, [])
    assert np.array_equal(out, img) and out is not img


def test_blur_faces_rgba_keeps_alpha(rgba, fake_faces):
    out = faces.blur_faces(rgba, sensibilidade=100, tipo="solido")
    assert np.array_equal(out[..., 3], rgba[..., 3])


def test_detect_without_model_raises(rgb, monkeypatch, tmp_path):
    monkeypatch.setattr(faces, "models_dir", lambda: tmp_path)
    with pytest.raises(bg.ModelNotDownloadedError):
        faces.detect_faces(rgb)


def test_invalid_blur_type(rgb):
    with pytest.raises(ValueError):
        faces.blur_boxes(rgb, [(0, 0, 0.5, 0.5, 1.0)], tipo="xyz")


def test_operations_registered(rgb, fake_matte, fake_faces):
    for key in ("desfocar_fundo", "desfocar_rostos"):
        step = ops.get_operation(key).new_step()
        assert ops.apply_step(rgb, step).shape == rgb.shape
        assert ops.get_operation(key).label in ops.describe_step(step)


@pytest.mark.skipif(not faces.model_available(), reason="modelo de rostos não baixado")
def test_real_detector_runs_on_blank_image(rgb):
    assert faces.detect_faces(np.zeros((300, 300, 3), np.uint8)) == []
