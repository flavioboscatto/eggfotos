"""Molduras: bordas de papel, rolo de filme, negativo, slide, filmadora VHS, TV de tubo…

Cada moldura é uma função `(rgb uint8, e, cor, texto) -> rgb uint8` registrada com @_register.
  e      0–2: fator da espessura/intensidade (1 = tamanho padrão da moldura; o usuário controla 0–100%)
  cor    (r, g, b) do papel/borda quando a moldura usa cor (senão ignorado)
  texto  legenda / data / marcação de borda, quando a moldura usa texto (senão ignorado)

Tamanhos são relativos ao menor lado da foto (regra 3 do projeto): a prévia reduzida fica igual
ao resultado final. Nada é aleatório sem semente fixa (regra 9). A saída é sempre RGB opaca:
fotos com transparência são achatadas sobre branco antes (a moldura precisa de um fundo).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .blur import _hex_to_rgb
from .image_io import flatten_alpha
from .imgutils import split_alpha

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class FrameSpec:
    key: str
    label: str
    group: str                       # "Papel e galeria" | "Cinema e filme" | "Eletrônicos"
    func: Callable[[np.ndarray, float, RGB, str], np.ndarray]
    cor_padrao: RGB | None           # None = a moldura não usa cor
    usa_texto: bool = False
    texto_dica: str = ""
    descricao: str = ""


FRAMES: dict[str, FrameSpec] = {}


def _register(key: str, label: str, group: str, cor: RGB | None = None, texto: bool = False,
              dica: str = "", desc: str = ""):
    def deco(func):
        FRAMES[key] = FrameSpec(key, label, group, func, cor, texto, dica, desc)
        return func
    return deco


def frame_keys() -> list[str]:
    return list(FRAMES)


def frame_label(key: str) -> str:
    return FRAMES[key].label


def apply_frame(img: np.ndarray, nome: str, espessura: float = 50, cor: str = "#ffffff",
                cor_padrao: bool = True, texto: str = "") -> np.ndarray:
    """Aplica a moldura `nome`. espessura 0–100 (50 = padrão). `cor_padrao`: ignora `cor` e usa a
    cor própria da moldura. Devolve RGB; o tamanho aumenta nas molduras de borda."""
    if nome not in FRAMES:
        raise ValueError(f"Moldura desconhecida: {nome}")
    spec = FRAMES[nome]
    rgb, alpha = split_alpha(img)
    if alpha is not None:
        rgb = flatten_alpha(img)
    rgb = np.ascontiguousarray(rgb)
    e = float(np.clip(espessura, 0, 100)) / 50.0
    color = spec.cor_padrao if (cor_padrao or spec.cor_padrao is None) else _hex_to_rgb(cor)
    return spec.func(rgb, e, color if color is not None else (0, 0, 0), (texto or "").strip())


# =================================================================== ferramentas

def _u(rgb: np.ndarray) -> int:
    return min(rgb.shape[:2])


def _px(frac: float, u: int, minimum: int = 1) -> int:
    return max(minimum, int(round(frac * u)))


def _font(size: float) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=max(6, int(round(size))))


def _lum(c: RGB) -> float:
    return (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255.0


def _canvas(h: int, w: int, color: RGB) -> np.ndarray:
    out = np.empty((h, w, 3), np.uint8)
    out[...] = color
    return out


def _rounded_mask(h: int, w: int, r: float, ss: int = 3) -> np.ndarray:
    """Máscara float 0–1 de um retângulo com cantos arredondados (suavizada)."""
    big = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(big).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), radius=max(0, r * ss), fill=255)
    return np.asarray(big.resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0


def _shadow(canvas: np.ndarray, x: int, y: int, w: int, h: int, blur: float, dy: float,
            opacity: float, radius: float = 0) -> None:
    """Sombra suave embaixo de um retângulo (altera `canvas` no lugar)."""
    H, W = canvas.shape[:2]
    m = np.zeros((H, W), np.float32)
    rm = _rounded_mask(h, w, radius) if radius > 0 else np.ones((h, w), np.float32)
    ys, xs = int(round(y + dy)), x
    y0, y1, x0, x1 = max(0, ys), min(H, ys + h), max(0, xs), min(W, xs + w)
    if y1 > y0 and x1 > x0:
        m[y0:y1, x0:x1] = rm[y0 - ys:y1 - ys, x0 - xs:x1 - xs]
    m = cv2.GaussianBlur(m, (0, 0), max(0.5, blur)) * opacity
    canvas[...] = (canvas.astype(np.float32) * (1.0 - m[..., None])).astype(np.uint8)


def _paste(canvas: np.ndarray, img: np.ndarray, x: int, y: int, mask: np.ndarray | None = None) -> None:
    h, w = img.shape[:2]
    H, W = canvas.shape[:2]
    y0, y1, x0, x1 = max(0, y), min(H, y + h), max(0, x), min(W, x + w)
    if y1 <= y0 or x1 <= x0:
        return
    src = img[y0 - y:y1 - y, x0 - x:x1 - x]
    if mask is None:
        canvas[y0:y1, x0:x1] = src
    else:
        m = mask[y0 - y:y1 - y, x0 - x:x1 - x, None]
        canvas[y0:y1, x0:x1] = (canvas[y0:y1, x0:x1] * (1 - m) + src * m + 0.5).astype(np.uint8)


def _paper_noise(canvas: np.ndarray, strength: float = 3.0, seed: int = 3) -> np.ndarray:
    """Textura sutil de papel (determinística)."""
    h, w = canvas.shape[:2]
    n = np.random.default_rng(seed).standard_normal((h // 2 + 1, w // 2 + 1)).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(canvas.astype(np.float32) + n[..., None] * strength, 0, 255).astype(np.uint8)


def _text(arr: np.ndarray, xy: tuple[float, float], txt: str, size: float, fill: RGB,
          anchor: str = "la", pixel: int = 0, shadow: RGB | None = None) -> None:
    """Escreve texto no array (no lugar). pixel>1: texto 'pixelado' (estilo display antigo)."""
    if not txt:
        return
    if pixel > 1:
        small_size = max(6, size / pixel)
        f = _font(small_size)
        l, t, r, b = f.getbbox(txt, anchor="la")
        wd, ht = int(r - l) + 4, int(b - t) + 4
        layer = Image.new("L", (wd, ht), 0)
        ImageDraw.Draw(layer).text((2 - l, 2 - t), txt, font=f, fill=255, anchor="la")
        layer = layer.resize((wd * pixel, ht * pixel), Image.NEAREST)
        m = np.asarray(layer, dtype=np.float32) / 255.0
        x, y = xy
        if anchor[0] == "m":
            x -= m.shape[1] / 2
        elif anchor[0] == "r":
            x -= m.shape[1]
        if anchor[1] == "m":
            y -= m.shape[0] / 2
        x, y = int(round(x)), int(round(y))
        if shadow is not None:
            sh = np.roll(np.roll(m, pixel, axis=0), pixel, axis=1)
            _blend(arr, sh, x, y, shadow)
        _blend(arr, m, x, y, fill)
        return
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    f = _font(size)
    if shadow is not None:
        d.text((xy[0] + max(1, size / 14), xy[1] + max(1, size / 14)), txt, font=f, fill=shadow, anchor=anchor)
    d.text(xy, txt, font=f, fill=fill, anchor=anchor)
    arr[...] = np.asarray(img)


def _blend(arr: np.ndarray, mask: np.ndarray, x: int, y: int, color: RGB) -> None:
    H, W = arr.shape[:2]
    h, w = mask.shape
    y0, y1, x0, x1 = max(0, y), min(H, y + h), max(0, x), min(W, x + w)
    if y1 <= y0 or x1 <= x0:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x, None]
    arr[y0:y1, x0:x1] = (arr[y0:y1, x0:x1] * (1 - m) + np.array(color, np.float32) * m + 0.5).astype(np.uint8)


def _scale(color: RGB, k: float) -> RGB:
    return tuple(int(np.clip(c * k, 0, 255)) for c in color)  # type: ignore[return-value]


def _vignette(rgb: np.ndarray, strength: float) -> np.ndarray:
    h, w = rgb.shape[:2]
    y = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    r = np.sqrt(x * x + y * y) / np.float32(np.sqrt(2))
    m = 1.0 - strength * np.clip((r - 0.35) / 0.65, 0, 1) ** 1.8
    return np.clip(rgb.astype(np.float32) * m[..., None] + 0.5, 0, 255).astype(np.uint8)


# ================================================================ papel e galeria

@_register("simples", "Borda simples", "Papel e galeria", (255, 255, 255),
           desc="Borda lisa em volta da foto.")
def _simples(rgb, e, cor, texto):
    t = _px(0.04 * e, _u(rgb), 0)
    if t == 0:
        return rgb.copy()
    h, w = rgb.shape[:2]
    out = _canvas(h + 2 * t, w + 2 * t, cor)
    _paste(out, rgb, t, t)
    return out


@_register("passepartout", "Passe-partout", "Papel e galeria", (245, 242, 235),
           desc="Papel largo de galeria, com leve sombra interna.")
def _passepartout(rgb, e, cor, texto):
    u, (h, w) = _u(rgb), rgb.shape[:2]
    t = _px(0.09 * e, u, 0)
    out = _canvas(h + 2 * t, w + 2 * t, cor)
    out = _paper_noise(out, 1.6)
    _shadow(out, t, t, w, h, blur=max(1.0, 0.006 * u), dy=0.003 * u, opacity=0.45)
    _paste(out, rgb, t, t)
    ln = _px(0.0015, u)
    cv2.rectangle(out, (t - ln, t - ln), (t + w + ln - 1, t + h + ln - 1), _scale(cor, 0.72), ln)
    return out


@_register("polaroid", "Polaroid", "Papel e galeria", (250, 248, 243), texto=True,
           dica="Legenda embaixo (opcional)", desc="Foto instantânea com a margem larga embaixo.")
def _polaroid(rgb, e, cor, texto):
    u, (h, w) = _u(rgb), rgb.shape[:2]
    t = _px(0.045 * e, u, 0)
    bottom = int(t * 4.2) if t else 0
    H, W = h + t + bottom, w + 2 * t
    out = _paper_noise(_canvas(H, W, cor), 1.4)
    _shadow(out, t, t, w, h, blur=max(1.0, 0.004 * u), dy=0.002 * u, opacity=0.25)
    _paste(out, rgb, t, t)
    if texto and bottom:
        _text(out, (W / 2, t + h + bottom * 0.5), texto, bottom * 0.30, (70, 70, 72), anchor="mm")
    return out


@_register("galeria", "Galeria (preto e dourado)", "Papel e galeria", (22, 22, 24),
           desc="Moldura escura com filete dourado e passe-partout claro.")
def _galeria(rgb, e, cor, texto):
    u, (h, w) = _u(rgb), rgb.shape[:2]
    outer = _px(0.05 * e, u, 0)
    gold = _px(0.0045, u, 2) if outer else 0
    mat = _px(0.05 * e, u, 0)
    pad = outer + gold + mat
    out = _canvas(h + 2 * pad, w + 2 * pad, cor)
    if gold:
        cv2.rectangle(out, (outer, outer), (outer + 2 * (gold + mat) + w - 1, outer + 2 * (gold + mat) + h - 1),
                      (201, 165, 82), -1)
    mx = outer + gold
    cv2.rectangle(out, (mx, mx), (mx + 2 * mat + w - 1, mx + 2 * mat + h - 1), (238, 235, 228), -1)
    _shadow(out, pad, pad, w, h, blur=max(1.0, 0.004 * u), dy=0.002 * u, opacity=0.35)
    _paste(out, rgb, pad, pad)
    return out


@_register("dupla", "Borda dupla", "Papel e galeria", (255, 255, 255),
           desc="Borda com um fio fino desenhado por dentro.")
def _dupla(rgb, e, cor, texto):
    u, (h, w) = _u(rgb), rgb.shape[:2]
    t = _px(0.055 * e, u, 0)
    out = _canvas(h + 2 * t, w + 2 * t, cor)
    _paste(out, rgb, t, t)
    if t >= 4:
        fio = (40, 40, 44) if _lum(cor) > 0.5 else (235, 235, 235)
        d = int(t * 0.45)
        cv2.rectangle(out, (d, d), (out.shape[1] - d - 1, out.shape[0] - d - 1), fio, _px(0.0018, u))
    return out


@_register("cartao", "Cartão com sombra", "Papel e galeria", (236, 236, 238),
           desc="Foto de cantos arredondados flutuando sobre um fundo suave.")
def _cartao(rgb, e, cor, texto):
    u, (h, w) = _u(rgb), rgb.shape[:2]
    p = _px(0.07 * e, u, 0)
    H, W = h + 2 * p, w + 2 * p
    bg = np.empty((H, W, 3), np.float32)
    grad = np.linspace(1.03, 0.9, H, dtype=np.float32)[:, None, None]
    bg[...] = np.array(cor, np.float32)
    out = np.clip(bg * grad, 0, 255).astype(np.uint8)
    r = 0.04 * u
    _shadow(out, p, p, w, h, blur=max(1.0, 0.02 * u * max(e, 0.3)), dy=0.012 * u, opacity=0.38, radius=r)
    _paste(out, rgb, p, p, _rounded_mask(h, w, r))
    return out


@_register("fita", "Colada com fita", "Papel e galeria", (250, 248, 243),
           desc="Foto revelada, colada na parede com dois pedaços de fita.")
def _fita(rgb, e, cor, texto):
    u, (h, w) = _u(rgb), rgb.shape[:2]
    t = _px(0.03 * e, u, 0)
    m = _px(0.08, u)
    ph, pw = h + 2 * t, w + 2 * t
    H, W = ph + 2 * m, pw + 2 * m
    wall = _paper_noise(_canvas(H, W, (222, 217, 208)), 2.2, seed=9)
    _shadow(wall, m, m, pw, ph, blur=0.012 * u, dy=0.006 * u, opacity=0.4)
    print_ = _canvas(ph, pw, cor)
    _paste(print_, rgb, t, t)
    _paste(wall, print_, m, m)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    tw, th = 0.17 * u, 0.055 * u
    for cx, cy, ang in ((m + 0.02 * u, m + 0.01 * u, -38), (m + pw - 0.02 * u, m + 0.01 * u, 38)):
        a = np.deg2rad(ang)
        pts = [(cx + dx * np.cos(a) - dy * np.sin(a), cy + dx * np.sin(a) + dy * np.cos(a))
               for dx, dy in ((-tw / 2, -th / 2), (tw / 2, -th / 2), (tw / 2, th / 2), (-tw / 2, th / 2))]
        d.polygon(pts, fill=(236, 224, 170, 190))
    base = Image.fromarray(wall).convert("RGBA")
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))


@_register("album", "Papel de álbum antigo", "Papel e galeria", (247, 243, 232),
           desc="Papel creme com a borda irregular, sobre a mesa.")
def _album(rgb, e, cor, texto):
    u, (h, w) = _u(rgb), rgb.shape[:2]
    t = _px(0.05 * e, u, 0)
    m = _px(0.06, u)
    ph, pw = h + 2 * t, w + 2 * t
    H, W = ph + 2 * m, pw + 2 * m
    table = _paper_noise(_canvas(H, W, (52, 49, 46)), 3.0, seed=4)
    # papel com a borda "mordida" por um ruído suave
    dist = cv2.distanceTransform(np.pad(np.ones((ph - 2, pw - 2), np.uint8), 1), cv2.DIST_L2, 3)
    noise = cv2.GaussianBlur(np.random.default_rng(7).standard_normal((ph, pw)).astype(np.float32),
                             (0, 0), max(1.0, 0.004 * u))
    noise = noise / (np.abs(noise).max() + 1e-6)
    amp = max(1.0, 0.012 * u * e)
    paper_mask = np.clip((dist - (amp * (0.5 + 0.5 * noise))) * 2.0, 0, 1)
    sheet = _paper_noise(_canvas(ph, pw, cor), 2.0, seed=2)
    _paste(sheet, rgb, t, t)
    _shadow(table, m, m, pw, ph, blur=0.012 * u, dy=0.006 * u, opacity=0.55)
    _paste(table, sheet, m, m, paper_mask.astype(np.float32))
    return table


# ================================================================ cinema e filme

@_register("cinema", "Barras de cinema", "Cinema e filme", (0, 0, 0), texto=True,
           dica="Legenda no rodapé (opcional)", desc="Barras pretas em cima e embaixo, com legenda de filme.")
def _cinema(rgb, e, cor, texto):
    h, w = rgb.shape[:2]
    bar = _px(0.11 * e, h, 0)
    out = _canvas(h + 2 * bar, w, cor)
    _paste(out, rgb, 0, bar)
    if texto and bar:
        size = bar * 0.42
        _text(out, (w / 2, h + bar + bar * 0.5), texto, size, (245, 245, 235), anchor="mm", shadow=(0, 0, 0))
    return out


def _sprocket_holes(img: np.ndarray, x0: int, x1: int, y_center: float, hole_w: float, hole_h: float,
                    pitch: float, color: RGB, ss: int = 3) -> None:
    """Fila horizontal de furos de filme (retângulos arredondados) entre x0 e x1."""
    H, W = img.shape[:2]
    hh = int(round(hole_h)) + 2
    layer = Image.new("L", (W * ss, hh * ss + ss), 0)
    d = ImageDraw.Draw(layer)
    n = int((x1 - x0) // pitch) + 1
    start = x0 + ((x1 - x0) - (n - 1) * pitch) / 2 - hole_w / 2
    for i in range(n):
        xa = (start + i * pitch) * ss
        d.rounded_rectangle((xa, ss, xa + hole_w * ss, ss + hole_h * ss), radius=hole_w * 0.22 * ss, fill=255)
    m = np.asarray(layer.resize((W, hh + 1), Image.LANCZOS), dtype=np.float32) / 255.0
    _blend(img, m, 0, int(round(y_center - hole_h / 2)) - 1, color)


def _film_strip(rgb: np.ndarray, e: float, texto: str, negativo: bool) -> np.ndarray:
    h, w = rgb.shape[:2]
    k = max(0.45, e)
    bar = _px(0.16 * k, h)
    side = _px(0.30, w)
    gap = _px(0.025, h)
    W, H = w + 2 * (side + gap), h + 2 * bar
    base = (46, 24, 12) if negativo else (10, 10, 11)
    empty = (66, 36, 18) if negativo else (24, 24, 26)
    hole_col = (228, 206, 168) if negativo else (238, 238, 236)
    out = _canvas(H, W, base)
    # quadros vizinhos (vazios) para parecer uma tira
    fx = side + gap
    out[bar:bar + h, 0:side] = empty
    out[bar:bar + h, W - side:W] = empty
    photo = rgb
    if negativo:
        inv = 255.0 - rgb.astype(np.float32)
        neg = (inv / 255.0 * 0.78 + 0.10) * np.array([1.0, 0.63, 0.40], np.float32) * 255.0
        photo = np.clip(neg, 0, 255).astype(np.uint8)
    _paste(out, photo, fx, bar)
    hole_h, hole_w = bar * 0.40, bar * 0.62
    pitch = hole_w * 1.95
    for yc in (bar * 0.5 - 0.0, H - bar * 0.5):
        _sprocket_holes(out, 0, W, yc, hole_w, hole_h, pitch, hole_col)
    edge = (255, 176, 46) if not negativo else (255, 196, 110)
    tsize = bar * 0.19
    marca = texto or "EGGFILM 400"
    # texto de borda: fabricante em cima, numeração embaixo (entre os furos e a foto)
    ytop = bar - tsize * 0.95
    ybot = H - bar + tsize * 0.15
    step = pitch * 3
    xx = pitch * 0.8
    n = 0
    while xx < W:
        _text(out, (xx, ytop), marca, tsize, edge)
        num = 10 + n
        _text(out, (xx + step * 0.1, ybot), f"{num}  >{num}A", tsize, edge)
        xx += step * 1.9
        n += 1
    return out


@_register("rolo35", "Rolo de filme 35 mm", "Cinema e filme", None, texto=True,
           dica="Marca na borda (padrão: EGGFILM 400)", desc="Tira de filme preta com furos e numeração.")
def _rolo35(rgb, e, cor, texto):
    return _film_strip(rgb, e, texto, negativo=False)


@_register("negativo35", "Negativo colorido 35 mm", "Cinema e filme", None, texto=True,
           dica="Marca na borda (padrão: EGGFILM 400)", desc="Fotografia como negativo laranja numa tira de filme.")
def _negativo35(rgb, e, cor, texto):
    return _film_strip(rgb, e, texto, negativo=True)


@_register("slide", "Slide 35 mm (diapositivo)", "Cinema e filme", (240, 238, 230), texto=True,
           dica="Rótulo no canto (ex.: 12A)", desc="Moldurinha de papelão dos slides antigos.")
def _slide(rgb, e, cor, texto):
    u, (h, w) = _u(rgb), rgb.shape[:2]
    t = _px(0.07 * e, u, 0)
    m = _px(0.04, u)
    H0, W0 = h + 2 * t, w + 2 * t
    H, W = H0 + 2 * m, W0 + 2 * m
    out = _paper_noise(_canvas(H, W, (40, 40, 43)), 2.0, seed=5)
    mount = _paper_noise(_canvas(H0, W0, cor), 1.6, seed=8)
    _paste(mount, rgb, t, t)
    ln = _px(0.0015, u)
    cv2.rectangle(mount, (t - ln, t - ln), (t + w + ln - 1, t + h + ln - 1), _scale(cor, 0.6), ln)
    if t >= 8:
        tx = t * 0.42
        _text(mount, (t * 0.9, H0 - t * 0.5), texto or "12", tx, (150, 80, 70), anchor="lm")
        _text(mount, (W0 - t * 0.9, t * 0.5), "EGGFILM", tx * 0.9, (120, 120, 118), anchor="rm")
        _text(mount, (W0 - t * 0.9, H0 - t * 0.5), "12A", tx * 0.9, (150, 80, 70), anchor="rm")
    r = 0.05 * u
    _shadow(out, m, m, W0, H0, blur=0.012 * u, dy=0.006 * u, opacity=0.6, radius=r)
    _paste(out, mount, m, m, _rounded_mask(H0, W0, r))
    return out


@_register("super8", "Filme Super 8", "Cinema e filme", None,
           desc="Quadro de filme de câmera antiga, com furos de um lado só.")
def _super8(rgb, e, cor, texto):
    u, (h, w) = _u(rgb), rgb.shape[:2]
    k = max(0.45, e)
    left = _px(0.13 * k, w)
    right = _px(0.035 * k, w)
    vb = _px(0.04 * k, h)
    H, W = h + 2 * vb, w + left + right
    out = _canvas(H, W, (12, 11, 11))
    r = 0.04 * u
    shot = _vignette(rgb, 0.28)
    _paste(out, shot, left, vb, _rounded_mask(h, w, r))
    # furos verticais na faixa da esquerda
    hole_w, hole_h = left * 0.46, left * 0.34
    pitch = hole_h * 2.1
    n = int(H // pitch)
    y0 = (H - (n - 1) * pitch) / 2
    ss = 3
    layer = Image.new("L", (left * ss, H * ss), 0)
    d = ImageDraw.Draw(layer)
    cx = left * 0.5
    for i in range(n):
        yc = y0 + i * pitch
        d.rounded_rectangle(((cx - hole_w / 2) * ss, (yc - hole_h / 2) * ss, (cx + hole_w / 2) * ss,
                             (yc + hole_h / 2) * ss), radius=hole_h * 0.25 * ss, fill=255)
    m = np.asarray(layer.resize((left, H), Image.LANCZOS), dtype=np.float32) / 255.0
    _blend(out, m, 0, 0, (236, 234, 228))
    # texto vertical na faixa da direita
    if right >= 8:
        tl = Image.new("L", (H, right), 0)
        ImageDraw.Draw(tl).text((H / 2, right / 2), "SUPER 8   EGGFILM", font=_font(right * 0.6), fill=255, anchor="mm")
        tm = np.asarray(tl.rotate(90, expand=True), dtype=np.float32) / 255.0
        _blend(out, tm, W - right, 0, (255, 176, 46))
    return out


# =================================================================== eletrônicos

@_register("vhs", "Filmadora VHS (REC)", "Eletrônicos", None, texto=True,
           dica="Data/hora na tela (padrão: data de hoje)",
           desc="Visor de filmadora antiga: REC, PLAY, bateria, data e linhas de varredura.")
def _vhs(rgb, e, cor, texto):
    u, (h, w) = _u(rgb), rgb.shape[:2]
    out = rgb.copy()
    # visual de fita: suaviza, separa levemente o vermelho, linhas de varredura e vinheta
    out = cv2.GaussianBlur(out, (0, 0), max(0.4, 0.0008 * max(h, w)))
    shift = max(1, int(round(0.0015 * w * min(e, 1.5))))
    out[..., 0] = np.roll(out[..., 0], shift, axis=1)
    rows = (np.arange(h) // max(1, int(round(0.0022 * u))) % 2).astype(np.float32)
    out = np.clip(out.astype(np.float32) * (1.0 - 0.13 * rows)[:, None, None] + 0.5, 0, 255).astype(np.uint8)
    out = _vignette(out, 0.30)
    white, red = (245, 245, 245), (232, 40, 40)
    shade = (0, 0, 0)
    size = 0.052 * u * max(0.6, e)
    pad = int(0.055 * u)
    px = max(2, int(round(size / 18)))            # tamanho do "pixel" do texto
    lw = max(2, int(0.0045 * u))
    # cantos do visor
    L = int(0.07 * u)
    for (cx, cy, sx, sy) in ((pad, pad, 1, 1), (w - pad, pad, -1, 1), (pad, h - pad, 1, -1), (w - pad, h - pad, -1, -1)):
        cv2.line(out, (cx, cy), (cx + sx * L, cy), white, lw, cv2.LINE_AA)
        cv2.line(out, (cx, cy), (cx, cy + sy * L), white, lw, cv2.LINE_AA)
    inset = int(pad * 1.7)
    # REC com bolinha
    r = int(size * 0.28)
    cv2.circle(out, (inset + r, inset + int(size * 0.5)), r, red, -1, cv2.LINE_AA)
    _text(out, (inset + 3 * r, inset), "REC", size, white, pixel=px, shadow=shade)
    # bateria (canto superior direito)
    bw, bh = int(size * 1.5), int(size * 0.72)
    bx, by = w - inset - bw, inset + int(size * 0.1)
    cv2.rectangle(out, (bx, by), (bx + bw, by + bh), white, lw)
    cv2.rectangle(out, (bx + bw, by + bh // 3), (bx + bw + lw * 2, by + 2 * bh // 3), white, -1)
    seg = (bw - 4 * lw) // 3
    for i in range(3):
        cv2.rectangle(out, (bx + 2 * lw + i * (seg + lw // 2), by + 2 * lw),
                      (bx + 2 * lw + i * (seg + lw // 2) + seg, by + bh - 2 * lw), white, -1)
    _text(out, (bx - int(size * 0.5), inset), "SP", size, white, anchor="ra", pixel=px, shadow=shade)
    # PLAY e data
    tri = int(size * 0.5)
    ty = h - inset - int(size * 0.5)
    cv2.fillPoly(out, [np.array([(inset, ty - tri), (inset, ty + tri), (inset + int(tri * 1.6), ty)], np.int32)], white, cv2.LINE_AA)
    _text(out, (inset + int(tri * 2.3), ty - int(size * 0.5)), "PLAY", size, white, pixel=px, shadow=shade)
    stamp = (texto or date.today().strftime("%b %d %Y")).upper()
    _text(out, (w - inset, ty - int(size * 0.5)), stamp, size, white, anchor="ra", pixel=px, shadow=shade)
    # mira no centro
    c, a = (w // 2, h // 2), int(0.03 * u)
    cv2.line(out, (c[0] - a, c[1]), (c[0] + a, c[1]), white, max(1, lw // 2), cv2.LINE_AA)
    cv2.line(out, (c[0], c[1] - a), (c[0], c[1] + a), white, max(1, lw // 2), cv2.LINE_AA)
    return out


@_register("crt", "TV de tubo (antiga)", "Eletrônicos", None,
           desc="Tela de TV antiga: cantos arredondados, linhas de varredura e reflexo.")
def _crt(rgb, e, cor, texto):
    u, (h, w) = _u(rgb), rgb.shape[:2]
    p = _px(0.06 * max(e, 0.3), u)
    H, W = h + 2 * p, w + 2 * p
    bezel = np.empty((H, W, 3), np.float32)
    bezel[...] = (34, 34, 37)
    gy = np.linspace(1.15, 0.8, H, dtype=np.float32)[:, None, None]
    out = np.clip(bezel * gy, 0, 255).astype(np.uint8)
    r = 0.07 * u
    screen = rgb.astype(np.float32)
    rows = (np.arange(h) // max(1, int(round(0.0025 * u))) % 2).astype(np.float32)
    screen *= (1.0 - 0.16 * rows)[:, None, None]
    screen = np.clip(screen * np.array([0.97, 1.0, 1.04], np.float32), 0, 255).astype(np.uint8)
    screen = _vignette(screen, 0.42)
    # reflexo suave no canto superior esquerdo
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    glow = np.clip(1.0 - (xx / w * 1.2 + yy / h * 1.6), 0, 1) ** 2 * 0.14
    screen = np.clip(screen.astype(np.float32) + glow[..., None] * 255, 0, 255).astype(np.uint8)
    _shadow(out, p, p, w, h, blur=0.012 * u, dy=0, opacity=0.7, radius=r)
    _paste(out, screen, p, p, _rounded_mask(h, w, r))
    return out
