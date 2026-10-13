"""Filtros predefinidos.

Para criar um filtro novo, basta escrever uma função `(rgb uint8) -> rgb uint8`
e decorá-la com @_register("chave", "Nome exibido"). Ele aparece sozinho no
painel de miniaturas e no processamento em lote.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .adjustments import adjust
from .imgutils import (
    gray3, luminance, merge_alpha, rel_odd, rel_sigma, split_alpha, to_float, to_uint8, unsharp, vignette,
)


@dataclass(frozen=True)
class FilterSpec:
    key: str
    label: str
    func: Callable[[np.ndarray], np.ndarray]


FILTERS: dict[str, FilterSpec] = {}


def _register(key: str, label: str):
    def deco(func):
        FILTERS[key] = FilterSpec(key, label, func)
        return func
    return deco


def filter_keys() -> list[str]:
    return list(FILTERS)


def filter_label(key: str) -> str:
    return FILTERS[key].label


def apply_filter(img: np.ndarray, nome: str, intensidade: float = 100) -> np.ndarray:
    """Aplica um filtro misturado à original conforme a intensidade (0–100)."""
    if nome not in FILTERS:
        raise ValueError(f"Filtro desconhecido: {nome}")
    t = float(np.clip(intensidade / 100.0, 0.0, 1.0))
    if nome == "original" or t == 0.0:
        return img.copy()
    rgb, alpha = split_alpha(img)
    rgb = np.ascontiguousarray(rgb)
    out = FILTERS[nome].func(rgb)
    if t < 1.0:
        out = cv2.addWeighted(rgb, 1.0 - t, out, t, 0)
    return merge_alpha(out, alpha)


# ---------------------------------------------------------------- auxiliares

def _clahe_l(rgb: np.ndarray, clip: float) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


_SEPIA = np.array(
    [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]], dtype=np.float32
)


def _sepia(rgb: np.ndarray) -> np.ndarray:
    return to_uint8(to_float(rgb) @ _SEPIA.T)


# ---------------------------------------------------- auxiliares (vintage)

def _grain(rgb: np.ndarray, amount: float, size: float = 0.0012, seed: int = 7) -> np.ndarray:
    """Grão de filme (determinístico e relativo ao tamanho: prévia = resultado final).

    `amount` ~ desvio do ruído (0–1); mais forte nos tons médios que nas pontas.
    """
    h, w = rgb.shape[:2]
    cell = max(1.0, size * max(h, w))
    sh, sw = max(2, int(round(h / cell))), max(2, int(round(w / cell)))
    noise = np.random.default_rng(seed).standard_normal((sh, sw)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    x = to_float(rgb)
    lum = luminance(x)
    weight = 1.0 - np.abs(lum - 0.5) * 1.2
    return to_uint8(x + (noise * amount * weight)[..., None])


def _curves(rgb: np.ndarray, r=None, g=None, b=None) -> np.ndarray:
    """Curvas por canal. Cada curva é uma lista de pontos (entrada, saída) em 0–1."""
    xs = np.arange(256, dtype=np.float32) / 255.0
    out = rgb.copy()
    for i, pts in enumerate((r, g, b)):
        if pts is None:
            continue
        px, py = zip(*pts)
        lut = to_uint8(np.interp(xs, px, py))
        out[..., i] = cv2.LUT(np.ascontiguousarray(rgb[..., i]), lut)
    return out


def _duotone(rgb: np.ndarray, shadow, highlight) -> np.ndarray:
    """Mapeia a luminância para um degradê entre duas cores RGB (0–255)."""
    lum = luminance(to_float(rgb))[..., None]
    lo = np.array(shadow, dtype=np.float32) / 255.0
    hi = np.array(highlight, dtype=np.float32) / 255.0
    return to_uint8(lo + (hi - lo) * lum)


def _dust(rgb: np.ndarray, density: float = 1.0, seed: int = 11) -> np.ndarray:
    """Poeira e riscos verticais de filme antigo (posições relativas)."""
    h, w = rgb.shape[:2]
    out = rgb.copy()
    rng = np.random.default_rng(seed)
    m = max(h, w)
    n_spots = int(40 * density)
    for _ in range(n_spots):
        cx, cy = int(rng.random() * w), int(rng.random() * h)
        r = max(1, int(m * rng.uniform(0.0005, 0.0022)))
        c = (235, 232, 222) if rng.random() < 0.7 else (25, 22, 20)
        cv2.circle(out, (cx, cy), r, c, -1, cv2.LINE_AA)
    for _ in range(int(5 * density)):
        x0 = int(rng.random() * w)
        y0 = int(rng.random() * h * 0.7)
        y1 = y0 + int(h * rng.uniform(0.1, 0.4))
        cv2.line(out, (x0, y0), (x0 + int(rng.normal(0, 2)), y1),
                 (225, 222, 212), max(1, int(m * 0.0006)), cv2.LINE_AA)
    return cv2.addWeighted(rgb, 0.45, out, 0.55, 0)


# ------------------------------------------------------------------- filtros

@_register("original", "Original")
def _original(rgb):
    return rgb.copy()


@_register("pb", "Preto e branco")
def _pb(rgb):
    return gray3(rgb)


@_register("noir", "Noir")
def _noir(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    out = adjust(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB), contraste=30)
    return vignette(out, 0.5)


@_register("sepia", "Sépia")
def _sepia_f(rgb):
    return _sepia(rgb)


@_register("vintage", "Vintage")
def _vintage(rgb):
    x = cv2.addWeighted(rgb, 0.5, _sepia(rgb), 0.5, 0)
    x = to_uint8(to_float(x) * 0.85 + 0.08)
    x = adjust(x, temperatura=25, saturacao=-10)
    return vignette(x, 0.35)


@_register("quente", "Quente")
def _quente(rgb):
    return adjust(rgb, temperatura=50, saturacao=8)


@_register("frio", "Frio")
def _frio(rgb):
    return adjust(rgb, temperatura=-50)


@_register("vivido", "Vívido")
def _vivido(rgb):
    return adjust(rgb, saturacao=40, contraste=15)


@_register("desbotado", "Desbotado")
def _desbotado(rgb):
    x = to_uint8(to_float(rgb) * 0.8 + 0.12)
    return adjust(x, saturacao=-30)


@_register("dramatico", "Dramático")
def _dramatico(rgb):
    return adjust(_clahe_l(rgb, 2.5), saturacao=-15, contraste=10)


@_register("hdr", "HDR")
def _hdr(rgb):
    x = unsharp(rgb, amount=0.8, sigma=rel_sigma(rgb, 0.02))
    return adjust(x, saturacao=15)


@_register("lomo", "Lomo")
def _lomo(rgb):
    return vignette(adjust(rgb, saturacao=30, contraste=25), 0.6)


@_register("suave", "Brilho suave")
def _suave(rgb):
    a = to_float(rgb)
    b = to_float(cv2.GaussianBlur(rgb, (0, 0), rel_sigma(rgb, 0.01)))
    screen = 1.0 - (1.0 - a) * (1.0 - b)
    return to_uint8(a * 0.5 + screen * 0.5)


@_register("nitido", "Nítido")
def _nitido(rgb):
    return unsharp(rgb, amount=1.0, sigma=rel_sigma(rgb, 0.0008, 0.6))


@_register("vinheta", "Vinheta")
def _vinheta(rgb):
    return vignette(rgb, 0.6)


@_register("esboco", "Esboço a lápis")
def _esboco(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(255 - g, (0, 0), rel_sigma(rgb, 0.006))
    sketch = cv2.divide(g, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)


@_register("cartoon", "Cartoon")
def _cartoon(rgb):
    h, w = rgb.shape[:2]
    # O filtro bilateral é caro: roda numa versão reduzida e volta ao tamanho.
    s = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    for _ in range(2):
        small = cv2.bilateralFilter(small, 9, 60, 60)
    color = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    g = cv2.medianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), rel_odd(rgb, 0.003))
    edges = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, rel_odd(rgb, 0.006, 5), 4
    )
    return cv2.bitwise_and(color, color, mask=edges)


@_register("negativo", "Negativo")
def _negativo(rgb):
    return 255 - rgb


# ------------------------------------------------- vintage e fotos antigas

@_register("foto_antiga", "Foto antiga")
def _foto_antiga(rgb):
    x = cv2.addWeighted(rgb, 0.4, _sepia(rgb), 0.6, 0)
    x = _curves(x, r=[(0, 0.10), (0.5, 0.56), (1, 0.96)],
                g=[(0, 0.08), (0.5, 0.50), (1, 0.92)],
                b=[(0, 0.07), (0.5, 0.42), (1, 0.82)])
    x = _grain(x, 0.045)
    return vignette(x, 0.45)


@_register("sepia_envelhecido", "Sépia envelhecido")
def _sepia_envelhecido(rgb):
    x = _duotone(gray3(rgb), (38, 22, 10), (238, 214, 166))
    x = _grain(x, 0.06, 0.0015)
    x = _dust(x, 1.0)
    return vignette(x, 0.6)


@_register("daguerreotipo", "Daguerreótipo")
def _daguerreotipo(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    x = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    x = cv2.GaussianBlur(x, (0, 0), rel_sigma(x, 0.0012, 0.4))
    x = _duotone(x, (18, 20, 24), (214, 214, 204))
    x = _grain(x, 0.04, 0.001)
    return vignette(x, 0.75)


@_register("polaroid", "Polaroid")
def _polaroid(rgb):
    x = _curves(rgb, r=[(0, 0.05), (0.5, 0.54), (1, 1.0)],
                g=[(0, 0.03), (0.5, 0.50), (1, 0.96)],
                b=[(0, 0.10), (0.5, 0.46), (1, 0.86)])
    x = adjust(x, contraste=-8, saturacao=-12, temperatura=10)
    x = _grain(x, 0.025)
    return vignette(x, 0.35)


@_register("anos70", "Anos 70")
def _anos70(rgb):
    x = adjust(rgb, temperatura=40, saturacao=-8, contraste=-5)
    x = _curves(x, r=[(0, 0.06), (1, 1.0)], g=[(0, 0.03), (0.5, 0.52), (1, 0.94)],
                b=[(0, 0.0), (0.5, 0.42), (1, 0.80)])
    x = _grain(x, 0.05)
    return vignette(x, 0.4)


@_register("kodachrome", "Kodachrome")
def _kodachrome(rgb):
    x = _curves(rgb, r=[(0, 0), (0.25, 0.22), (0.75, 0.82), (1, 1)],
                g=[(0, 0), (0.5, 0.49), (1, 0.97)],
                b=[(0, 0.02), (0.5, 0.45), (1, 0.92)])
    x = adjust(x, saturacao=22, contraste=18, temperatura=12)
    return _grain(x, 0.02, 0.001)


@_register("cross_process", "Cross process")
def _cross_process(rgb):
    x = _curves(rgb, r=[(0, 0), (0.4, 0.32), (0.75, 0.9), (1, 1)],
                g=[(0, 0.05), (0.5, 0.55), (1, 1)],
                b=[(0, 0.15), (0.5, 0.45), (1, 0.78)])
    return adjust(x, saturacao=18, contraste=15)


@_register("filme_pb", "Filme P&B clássico")
def _filme_pb(rgb):
    x = gray3(rgb)
    x = _curves(x, r=[(0, 0.04), (0.3, 0.24), (0.7, 0.78), (1, 0.97)],
                g=[(0, 0.04), (0.3, 0.24), (0.7, 0.78), (1, 0.97)],
                b=[(0, 0.04), (0.3, 0.24), (0.7, 0.78), (1, 0.97)])
    x = _grain(x, 0.07, 0.0013)
    return vignette(x, 0.4)


@_register("filme_8mm", "Filme 8mm")
def _filme_8mm(rgb):
    x = adjust(rgb, temperatura=20, saturacao=-18, contraste=8)
    x = _curves(x, g=[(0, 0.04), (1, 0.97)], b=[(0, 0.08), (1, 0.85)])
    x = _grain(x, 0.08, 0.0016)
    x = _dust(x, 1.5, seed=23)
    return vignette(x, 0.65)


@_register("papel_amarelado", "Papel amarelado")
def _papel_amarelado(rgb):
    x = cv2.addWeighted(rgb, 0.55, _sepia(rgb), 0.45, 0)
    x = to_uint8(to_float(x) * 0.78 + 0.16)
    paper = np.array([1.0, 0.93, 0.72], dtype=np.float32)
    x = to_uint8(to_float(x) * paper)
    x = _grain(x, 0.035, 0.002, seed=3)
    return vignette(x, 0.7)
