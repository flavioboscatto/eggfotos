"""Ajustes de luz e cor. Vão de -100 a 100 (0 = neutro), exceto os de ADJUSTMENT_RANGES."""
from __future__ import annotations

import cv2
import numpy as np

from .imgutils import luminance, merge_alpha, rel_sigma, split_alpha, to_float, to_uint8, unsharp

ADJUSTMENT_KEYS = (
    "exposicao", "brilho", "contraste", "realces", "sombras",
    "saturacao", "temperatura", "nitidez", "ruido",
)
ADJUSTMENT_RANGES = {"ruido": (0, 100)}  # faixa diferente de −100..100


def adjustment_range(key: str) -> tuple[int, int]:
    return ADJUSTMENT_RANGES.get(key, (-100, 100))

ADJUSTMENT_LABELS = {
    "exposicao": "Exposição",
    "brilho": "Brilho",
    "contraste": "Contraste",
    "realces": "Realces",
    "sombras": "Sombras",
    "saturacao": "Saturação",
    "temperatura": "Temperatura",
    "nitidez": "Nitidez",
    "ruido": "Remover granulação",
}


def neutral_params() -> dict[str, int]:
    return {k: 0 for k in ADJUSTMENT_KEYS}


def is_neutral(params: dict) -> bool:
    return all(float(params.get(k, 0)) == 0 for k in ADJUSTMENT_KEYS)


def denoise(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Reduz granulação/ruído (amount 0–100) preservando bordas.

    Luminância: Non-Local Means (preserva detalhe). Cor (Cr/Cb): suavização forte,
    pois o ruído de croma é o mais feio e o olho quase não vê detalhe de cor.
    O tamanho do ruído é em pixels reais: confira o efeito em 100% (a prévia reduzida
    tende a parecer mais limpa que o resultado final).
    """
    t = float(np.clip(amount, 0, 100)) / 100.0
    if t == 0:
        return rgb.copy()
    ycc = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2YCrCb)
    y = cv2.fastNlMeansDenoising(ycc[..., 0], None, h=2.0 + 16.0 * t,
                                 templateWindowSize=7, searchWindowSize=15)
    sigma = rel_sigma(rgb, 0.0015 + 0.004 * t, 0.5)
    chroma = cv2.GaussianBlur(ycc[..., 1:], (0, 0), sigma)
    ycc = np.dstack([y, chroma])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def adjust(img: np.ndarray, **params) -> np.ndarray:
    """Aplica os ajustes na ordem de um revelador de fotos."""
    unknown = set(params) - set(ADJUSTMENT_KEYS)
    if unknown:
        raise ValueError(f"Ajuste desconhecido: {', '.join(sorted(unknown))}")
    p = {k: float(np.clip(params.get(k, 0), *adjustment_range(k))) for k in ADJUSTMENT_KEYS}
    if is_neutral(p):
        return img.copy()

    rgb, alpha = split_alpha(img)
    if p["ruido"]:  # primeiro: a granulação não deve ser amplificada pelos outros ajustes
        rgb = denoise(rgb, p["ruido"])
    x = to_float(rgb)

    if p["exposicao"]:
        x *= 2.0 ** (p["exposicao"] / 50.0)  # ±2 stops
    if p["brilho"]:
        x += p["brilho"] / 100.0 * 0.3
    if p["contraste"]:
        f = 1.0 + p["contraste"] / 100.0
        x = (x - 0.5) * f + 0.5
    if p["sombras"] or p["realces"]:
        lum = luminance(np.clip(x, 0.0, 1.0))[..., None]
        if p["sombras"]:
            x += (p["sombras"] / 100.0 * 0.4) * (1.0 - lum) ** 2
        if p["realces"]:
            x += (p["realces"] / 100.0 * 0.4) * lum ** 2
    if p["saturacao"]:
        lum = luminance(x)[..., None]
        x = lum + (x - lum) * (1.0 + p["saturacao"] / 100.0)
    if p["temperatura"]:
        t = p["temperatura"] / 100.0
        x[..., 0] *= 1.0 + 0.12 * t
        x[..., 2] *= 1.0 - 0.12 * t

    out = to_uint8(x)
    if p["nitidez"] > 0:
        out = unsharp(out, amount=p["nitidez"] / 100.0 * 1.5, sigma=rel_sigma(out, 0.0004, 0.6))
    elif p["nitidez"] < 0:
        out = cv2.GaussianBlur(out, (0, 0), rel_sigma(out, 0.002 * (-p["nitidez"] / 100.0)))
    return merge_alpha(out, alpha)
