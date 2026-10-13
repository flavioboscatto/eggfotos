import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def rgb():
    """Imagem sintética 120x200 com gradiente e um quadrado colorido."""
    h, w = 120, 200
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([(x * 255 // w), (y * 255 // h), ((x + y) * 255 // (w + h))], axis=2).astype(np.uint8)
    img[40:80, 60:120] = (220, 40, 40)
    return img


@pytest.fixture
def rgba(rgb):
    a = np.full(rgb.shape[:2] + (1,), 255, np.uint8)
    a[:, :50] = 0
    return np.concatenate([rgb, a], axis=2)


@pytest.fixture
def fake_matte(monkeypatch):
    """Troca o modelo de IA por uma máscara fixa (quadrado central), sem rede nem download."""
    from eggfotos.core import background

    def _matte(img, modelo="equilibrado"):
        h, w = img.shape[:2]
        m = np.zeros((h, w), np.uint8)
        m[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 255
        return m

    monkeypatch.setattr(background, "compute_matte", _matte)
    return _matte


@pytest.fixture
def fake_faces(monkeypatch):
    """Dois 'rostos' fixos (um forte, um fraco), sem modelo nem download."""
    from eggfotos.core import faces

    boxes = [(0.1, 0.1, 0.2, 0.3, 0.95), (0.6, 0.5, 0.2, 0.3, 0.40)]
    monkeypatch.setattr(faces, "detect_faces", lambda img: list(boxes))
    return boxes
