"""Ferramenta de borrar: gaussiano, mosaico (pixelização) ou tarja sólida,
aplicados numa máscara (pincel) ou numa região (retângulo / elipse)."""
from __future__ import annotations

import cv2
import numpy as np

BLUR_TYPES = {"gaussiano": "Gaussiano", "mosaico": "Mosaico", "solido": "Tarja sólida"}
SHAPES = {"retangulo": "Retângulo", "elipse": "Elipse"}


def _hex_to_rgb(cor: str) -> tuple[int, int, int]:
    cor = cor.lstrip("#")
    if len(cor) != 6:
        raise ValueError(f"Cor inválida: #{cor}")
    return int(cor[0:2], 16), int(cor[2:4], 16), int(cor[4:6], 16)


def _effect(roi: np.ndarray, tipo: str, intensidade: float, ref_size: int, cor: str) -> np.ndarray:
    """Calcula o efeito sobre a região. Raios são relativos ao tamanho da foto
    inteira (ref_size), para prévia e resultado final ficarem iguais."""
    t = float(np.clip(intensidade, 1, 100)) / 100.0
    if tipo == "gaussiano":
        sigma = max(0.5, t * 0.03 * ref_size)
        return cv2.GaussianBlur(roi, (0, 0), sigma)
    if tipo == "mosaico":
        block = max(2, int(round(t * 0.05 * ref_size)))
        h, w = roi.shape[:2]
        small = cv2.resize(
            roi, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA
        )
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if tipo == "solido":
        out = np.empty_like(roi)
        out[..., :3] = _hex_to_rgb(cor)
        if roi.shape[2] == 4:
            out[..., 3] = roi[..., 3]
        return out
    raise ValueError(f"Tipo de desfoque inválido: {tipo}")


def blur_with_mask(
    img: np.ndarray,
    mask: np.ndarray,
    tipo: str = "gaussiano",
    intensidade: float = 50,
    cor: str = "#000000",
    cor_original: bool = False,
) -> np.ndarray:
    """Aplica o efeito onde mask > 0 (mask uint8 HxW, 0–255).

    `cor_original`: a tarja sólida usa a cor média da própria área (em vez de `cor`).

    Só processa o retângulo envolvente da máscara (com folga), o que deixa
    o pincel rápido mesmo em fotos grandes.
    """
    if mask.shape[:2] != img.shape[:2]:
        raise ValueError("A máscara deve ter o mesmo tamanho da imagem.")
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return img.copy()
    H, W = img.shape[:2]
    ref = max(H, W)
    if tipo == "solido" and cor_original:
        mean = img[..., :3][mask > 0].reshape(-1, 3).mean(axis=0)
        cor = "#{:02x}{:02x}{:02x}".format(*(int(round(c)) for c in mean))
    pad = int(0.1 * ref * float(np.clip(intensidade, 1, 100)) / 100.0) + 4
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)

    roi = img[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1].astype(np.float32) / 255.0
    if tipo == "gaussiano":  # borda suave só no gaussiano; mosaico/tarja ficam nítidos
        m = cv2.GaussianBlur(m, (0, 0), max(1.0, 0.002 * ref))
    effect = _effect(roi, tipo, intensidade, ref, cor)

    m = m[..., None]
    mixed = roi.astype(np.float32) * (1.0 - m) + effect.astype(np.float32) * m
    out = img.copy()
    out[y0:y1, x0:x1] = np.clip(mixed + 0.5, 0, 255).astype(np.uint8)
    return out


def region_mask(shape: tuple[int, ...], x: int, y: int, w: int, h: int, forma: str = "retangulo") -> np.ndarray:
    """Máscara uint8 de um retângulo ou elipse em pixels."""
    H, W = shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
    if forma == "retangulo":
        cv2.rectangle(mask, (x, y), (x + w - 1, y + h - 1), 255, thickness=-1)
    elif forma == "elipse":
        cv2.ellipse(mask, (x + w // 2, y + h // 2), (max(1, w // 2), max(1, h // 2)), 0, 0, 360, 255, -1)
    else:
        raise ValueError(f"Forma inválida: {forma}")
    return mask


def blur_region(
    img: np.ndarray,
    x: float, y: float, w: float, h: float,
    forma: str = "retangulo",
    tipo: str = "gaussiano",
    intensidade: float = 50,
    cor: str = "#000000",
    cor_original: bool = False,
) -> np.ndarray:
    """Borra uma região dada em coordenadas RELATIVAS (0–1), útil no lote."""
    H, W = img.shape[:2]
    mask = region_mask(img.shape, x * W, y * H, w * W, h * H, forma)
    return blur_with_mask(img, mask, tipo, intensidade, cor, cor_original)


def stroke_mask(shape: tuple[int, ...], points: list[tuple[float, float]], size: float) -> np.ndarray:
    """Converte um traço de pincel (lista de pontos em pixels) numa máscara."""
    H, W = shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    thickness = max(1, int(round(size)))
    pts = [(int(round(px)), int(round(py))) for px, py in points]
    if not pts:
        return mask
    cv2.circle(mask, pts[0], max(1, thickness // 2), 255, -1, cv2.LINE_AA)
    for a, b in zip(pts, pts[1:]):
        cv2.line(mask, a, b, 255, thickness, cv2.LINE_AA)
    return mask
