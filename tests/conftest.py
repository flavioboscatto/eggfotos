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


@pytest.fixture
def fake_sam(monkeypatch):
    """MobileSAM falso: devolve sempre um quadrado de 60×60 px em volta do 1º ponto."""
    from eggfotos.core import segment

    class _Enc:
        def run(self, _out, feeds):
            return [np.zeros((1, 256, 64, 64), np.float32)]

    class _Dec:
        def run(self, _out, feeds):
            H, W = (int(v) for v in feeds["orig_im_size"])
            s = max(H, W) and 1024 / max(H, W)
            x, y = feeds["point_coords"][0, 0] / s
            m = np.full((1, 1, H, W), -5.0, np.float32)
            m[0, 0, max(0, int(y) - 30):int(y) + 30, max(0, int(x) - 30):int(x) + 30] = 5.0
            return [m, np.ones((1, 1), np.float32)]

    monkeypatch.setattr(segment, "_session", lambda kind: _Enc() if kind == "encoder" else _Dec())
    return segment


@pytest.fixture
def fake_objects(monkeypatch, tmp_path):
    """SegFormer falso: devolve classes fixas (pessoa à esquerda, carro à direita, prédio no meio,
    céu em cima), sem modelo nem download. Marca o modelo como baixado."""
    from eggfotos.core import objects

    class _Sess:
        def run(self, _out, feeds):
            _, _, h, w = feeds["pixel_values"].shape
            hh, ww = h // 4, w // 4
            logits = np.full((1, 150, hh, ww), -5.0, np.float32)
            logits[0, 2, :hh // 4, :] = 5.0                          # céu (faixa superior)
            logits[0, 12, hh // 4:, : ww // 3] = 5.0                 # pessoa (terço esquerdo)
            logits[0, 1, hh // 4:, ww // 3: 2 * ww // 3] = 5.0       # prédio (terço do meio)
            logits[0, 20, hh // 4:, 2 * ww // 3:] = 5.0              # carro (terço direito)
            return [logits]

    monkeypatch.setattr(objects, "_session", lambda modelo: _Sess())
    monkeypatch.setattr(objects, "model_available", lambda modelo=objects.DEFAULT_MODEL: True)
    return objects


@pytest.fixture
def marks_tmp(monkeypatch, tmp_path):
    """Pasta de modelos de marca isolada (não toca em ~/.eggfotos/marcas)."""
    from eggfotos.core import watermark
    monkeypatch.setattr(watermark, "marks_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture(scope="session")
def qapp():
    """QApplication única da sessão (precisa ficar referenciada, senão o Qt aborta)."""
    pytest.importorskip("PyQt6")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app
