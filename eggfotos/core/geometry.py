"""Recorte, rotação, espelhamento e redimensionamento."""
from __future__ import annotations

import math

import cv2
import numpy as np

RATIO_CHOICES = ("1:1", "4:3", "3:2", "16:9", "5:4", "3:4", "2:3", "9:16", "4:5")


def parse_ratio(text: str | None) -> float | None:
    """'3:2' -> 1.5 ; 'livre'/None -> None."""
    if not text or text.lower() in ("livre", "original"):
        return None
    a, b = text.replace("x", ":").split(":")
    a, b = float(a), float(b)
    if a <= 0 or b <= 0:
        raise ValueError(f"Proporção inválida: {text}")
    return a / b


def downscale(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Reduz para caber em max_side. Retorna (imagem, escala aplicada)."""
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img, 1.0
    scale = max_side / s
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA), scale


def rotate90(img: np.ndarray, vezes: int = 1) -> np.ndarray:
    """Gira em múltiplos de 90° no sentido horário (vezes negativo = anti-horário)."""
    return np.ascontiguousarray(np.rot90(img, -int(vezes) % 4))


def rotate_angle(img: np.ndarray, angulo: int) -> np.ndarray:
    """Gira 90, 180 ou 270 graus no sentido horário."""
    if int(angulo) % 90 != 0:
        raise ValueError("Use endireitar() para ângulos que não são múltiplos de 90°.")
    return rotate90(img, int(angulo) // 90)


def flip(img: np.ndarray, direcao: str = "horizontal") -> np.ndarray:
    if direcao == "horizontal":
        return np.ascontiguousarray(img[:, ::-1])
    if direcao == "vertical":
        return np.ascontiguousarray(img[::-1])
    raise ValueError(f"Direção inválida: {direcao}")


def _max_inner_rect(w: float, h: float, angle_rad: float) -> tuple[float, float]:
    """Maior retângulo alinhado aos eixos dentro de um retângulo w×h girado."""
    if w <= 0 or h <= 0:
        return 0.0, 0.0
    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)
    sin_a, cos_a = abs(math.sin(angle_rad)), abs(math.cos(angle_rad))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


def straighten(img: np.ndarray, angulo: float) -> np.ndarray:
    """Gira por um ângulo livre (graus, positivo = anti-horário) e recorta
    as bordas vazias, como a ferramenta 'endireitar' dos editores."""
    angulo = float(angulo)
    if abs(angulo) < 1e-6:
        return img.copy()
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angulo, 1.0)
    rotated = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    wr, hr = _max_inner_rect(w, h, math.radians(angulo))
    wr, hr = int(min(w, wr)), int(min(h, hr))
    x0, y0 = (w - wr) // 2, (h - hr) // 2
    return np.ascontiguousarray(rotated[y0:y0 + hr, x0:x0 + wr])


def crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Recorte em pixels, limitado às bordas da imagem."""
    H, W = img.shape[:2]
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(W, int(round(x + w))), min(H, int(round(y + h)))
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise ValueError("Área de recorte vazia.")
    return np.ascontiguousarray(img[y0:y1, x0:x1])


def crop_relative(img: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    """Recorte com coordenadas relativas (0–1). Serve para o lote:
    a mesma área relativa vale para fotos de tamanhos diferentes."""
    H, W = img.shape[:2]
    return crop(img, x * W, y * H, w * W, h * H)


def crop_to_ratio(img: np.ndarray, proporcao: str, ancora: str = "centro") -> np.ndarray:
    """Recorta a maior área possível com a proporção pedida."""
    r = parse_ratio(proporcao)
    if r is None:
        return img.copy()
    H, W = img.shape[:2]
    if W / H > r:
        w, h = int(round(H * r)), H
    else:
        w, h = W, int(round(W / r))
    x = (W - w) // 2
    y = (H - h) // 2
    if ancora == "topo":
        y = 0
    elif ancora == "base":
        y = H - h
    elif ancora == "esquerda":
        x = 0
    elif ancora == "direita":
        x = W - w
    elif ancora != "centro":
        raise ValueError(f"Âncora inválida: {ancora}")
    return crop(img, x, y, w, h)


def resize(img: np.ndarray, largura: int, altura: int) -> np.ndarray:
    largura, altura = max(1, int(largura)), max(1, int(altura))
    H, W = img.shape[:2]
    interp = cv2.INTER_AREA if largura * altura < W * H else cv2.INTER_LANCZOS4
    return cv2.resize(img, (largura, altura), interpolation=interp)


def resize_long_side(img: np.ndarray, lado_maior: int, ampliar: bool = False) -> np.ndarray:
    """Redimensiona mantendo a proporção para que o maior lado tenha `lado_maior` px."""
    H, W = img.shape[:2]
    s = max(H, W)
    if s == lado_maior or (s < lado_maior and not ampliar):
        return img.copy()
    k = lado_maior / s
    return resize(img, round(W * k), round(H * k))
