"""Utilitários compartilhados pelas operações de imagem."""
from __future__ import annotations

import numpy as np
import cv2

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def split_alpha(img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Separa RGB e canal alpha (se existir)."""
    if img.ndim == 3 and img.shape[2] == 4:
        return img[..., :3], img[..., 3:]
    return img, None


def merge_alpha(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """Recoloca o canal alpha separado por split_alpha."""
    if alpha is None:
        return np.ascontiguousarray(rgb)
    if alpha.shape[:2] != rgb.shape[:2]:
        alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        alpha = alpha[..., None] if alpha.ndim == 2 else alpha
    return np.ascontiguousarray(np.concatenate([rgb, alpha], axis=2))


def to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) * (1.0 / 255.0)


def to_uint8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def luminance(x: np.ndarray) -> np.ndarray:
    """Luminância (float) de uma imagem RGB float."""
    return x @ _LUMA


def rel_sigma(img: np.ndarray, frac: float, minimum: float = 0.3) -> float:
    """Sigma proporcional ao tamanho da imagem.

    Usar raios relativos faz a prévia (imagem reduzida) e o resultado em
    resolução total ficarem visualmente iguais.
    """
    return max(minimum, frac * max(img.shape[0], img.shape[1]))


def rel_odd(img: np.ndarray, frac: float, minimum: int = 3) -> int:
    """Tamanho de kernel ímpar proporcional ao tamanho da imagem."""
    k = max(minimum, int(round(frac * max(img.shape[0], img.shape[1]))))
    return k if k % 2 == 1 else k + 1


def unsharp(img: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    """Máscara de nitidez (unsharp mask) em uint8."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def vignette(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Escurece as bordas. strength 0..1."""
    h, w = rgb.shape[:2]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    r = np.sqrt(x * x + y * y) / np.float32(np.sqrt(2.0))
    mask = 1.0 - strength * np.clip((r - 0.3) / 0.7, 0.0, 1.0) ** 1.6
    out = rgb.astype(np.float32) * mask[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def gray3(rgb: np.ndarray) -> np.ndarray:
    """Converte para tons de cinza mantendo 3 canais."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
