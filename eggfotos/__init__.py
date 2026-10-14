"""EggFotos — editor de fotos desktop com processamento em lote."""

__version__ = "0.1.0"
APP_NAME = "EggFotos"

import sys as _sys
from pathlib import Path as _Path

# pasta de recursos (ícones, logo): ao lado do código; dentro do app empacotado (PyInstaller) em _MEIPASS
ASSETS_DIR = _Path(getattr(_sys, "_MEIPASS", _Path(__file__).resolve().parents[1])) / "assets"
ICON_PATH = ASSETS_DIR / "icon" / "EggFotos.png"
