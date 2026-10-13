import numpy as np
import pytest

from eggfotos.core import segment


def test_without_models_raises(rgb, monkeypatch, tmp_path):
    monkeypatch.setattr(segment, "models_dir", lambda: tmp_path)
    segment._sessions.clear()
    assert not segment.model_available()
    with pytest.raises(segment.ModelNotDownloadedError):
        segment.embed(rgb)


def test_embed_resizes_to_1024_and_predict_maps_points(fake_sam):
    img = np.zeros((300, 400, 3), np.uint8)
    emb = segment.embed(img)
    assert emb.size == (300, 400) and abs(emb.scale - 1024 / 400) < 1e-9
    mask = segment.predict(emb, [(0.5, 0.5, 1)])   # centro da foto
    assert mask.shape == (300, 400) and mask.dtype == np.uint8
    assert mask[150, 200] == 255 and mask[10, 10] == 0
    ys, xs = np.nonzero(mask)
    assert abs((xs.min() + xs.max()) / 2 - 200) < 2 and abs((ys.min() + ys.max()) / 2 - 150) < 2


def test_predict_without_points_is_empty(fake_sam):
    emb = segment.embed(np.zeros((50, 60, 3), np.uint8))
    assert not segment.predict(emb, []).any()


def test_alpha_ignored(rgba, fake_sam):
    assert segment.select_object(rgba, [(0.5, 0.5, 1)]).shape == rgba.shape[:2]


@pytest.mark.skipif(not segment.model_available(), reason="modelos MobileSAM não baixados")
def test_real_model_selects_disc():
    import cv2
    img = np.full((600, 800, 3), (40, 90, 50), np.uint8)
    cv2.circle(img, (500, 250), 120, (220, 60, 60), -1)
    truth = np.zeros((600, 800), np.uint8)
    cv2.circle(truth, (500, 250), 120, 1, -1)
    m = segment.select_object(img, [(500 / 800, 250 / 600, 1)]) > 0
    t = truth > 0
    assert (m & t).sum() / (m | t).sum() > 0.9
