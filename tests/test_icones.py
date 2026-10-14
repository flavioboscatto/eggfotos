import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from eggfotos import ASSETS_DIR, ICON_PATH

RAIZ = Path(__file__).resolve().parents[1]


def _gerador():
    spec = importlib.util.spec_from_file_location("gerar_icones", RAIZ / "tools" / "gerar_icones.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_icon_files_exist_with_expected_sizes():
    from PIL import Image
    assert ICON_PATH.is_file() and ASSETS_DIR.is_dir()
    img = Image.open(ICON_PATH)
    assert img.size == (1024, 1024) and img.mode == "RGBA"
    for ext in ("ico", "icns"):
        assert (ASSETS_DIR / "icon" / f"EggFotos.{ext}").stat().st_size > 1000
    assert img.getpixel((0, 0))[3] == 0                      # cantos transparentes (formato arredondado)
    assert img.getpixel((512, 512))[3] == 255                # miolo opaco


def test_symbol_extraction_ignores_the_text_below():
    logo = ASSETS_DIR / "logo" / "eggfotos_logo.jpg"
    if not logo.is_file():
        pytest.skip("logo não está no projeto")
    g = _gerador()
    s = g.extrair_simbolo(logo)
    assert s.ndim == 2 and 0.0 <= s.min() and s.max() <= 1.0
    assert 0.8 < s.shape[1] / s.shape[0] < 1.2        # ovo + câmera, quase quadrado (sem o texto largo)


def test_squircle_is_symmetric_and_centered():
    g = _gerador()
    m = g.squircle(256, 200, ss=2)
    assert np.allclose(m, m[::-1, :], atol=0.02) and np.allclose(m, m[:, ::-1], atol=0.02)
    assert m[128, 128] == 1.0 and m[0, 0] == 0.0 and m[128, 20] == 0.0 and m[128, 40] == 1.0


def test_app_window_icon_loads(qapp):
    from PyQt6.QtGui import QIcon
    icon = QIcon(str(ICON_PATH))
    assert not icon.isNull() and not icon.pixmap(64, 64).isNull()
