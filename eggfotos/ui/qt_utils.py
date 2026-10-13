"""Conversões NumPy <-> Qt e pequenos utilitários de interface."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap


def to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        fmt, bpl = QImage.Format.Format_Grayscale8, w
    elif arr.shape[2] == 3:
        fmt, bpl = QImage.Format.Format_RGB888, 3 * w
    elif arr.shape[2] == 4:
        fmt, bpl = QImage.Format.Format_RGBA8888, 4 * w
    else:
        raise ValueError(f"Formato de array não suportado: {arr.shape}")
    # .copy() desacopla o QImage do buffer NumPy (que pode ser liberado depois)
    return QImage(arr.data, w, h, bpl, fmt).copy()


def to_qpixmap(arr: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(to_qimage(arr))


def fit_pixmap(arr: np.ndarray, size: int) -> QPixmap:
    """Pixmap reduzido para caber num quadrado `size` (para miniaturas)."""
    pm = to_qpixmap(arr)
    return pm.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                     Qt.TransformationMode.SmoothTransformation)


def debounce(parent, msec: int, slot) -> QTimer:
    """Timer de disparo único: chame .start() a cada evento; o slot só roda
    quando os eventos param por `msec` ms (evita recalcular a cada pixel do slider)."""
    t = QTimer(parent)
    t.setSingleShot(True)
    t.setInterval(msec)
    t.timeout.connect(slot)
    return t
