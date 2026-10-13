"""Remoção de objetos / marca d'água por preenchimento (inpainting).

Dois métodos, mesma interface (imagem + máscara uint8, 255 = apagar):
  - "telea" / "ns": cv2.inpaint, rápido e sem modelo; bom para marcas finas e fundos lisos.
  - "lama": rede LaMa (ONNX), bem melhor em texturas e áreas grandes; modelo baixado sob demanda.

Só a região da máscara (mais contexto) é processada, então é rápido mesmo em fotos grandes.
O LaMa roda em JANELAS de 512×512 pixels REAIS (sem reduzir a foto), em vários blocos se a marca
for grande; assim, quanto maior a resolução, mais detalhe o preenchimento herda. Depois o grão
da foto é recolocado no trecho preenchido, que sai liso demais.
Raios e margens são relativos ao tamanho da foto (regra 3). Fora da máscara a foto é
preservada pixel a pixel.
"""
from __future__ import annotations

import ssl
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .background import ModelNotDownloadedError, _rembg, models_dir
from .imgutils import merge_alpha, split_alpha

METHODS = {
    "telea": "Rápido — Telea (sem modelo)",
    "ns": "Rápido — Navier-Stokes (sem modelo)",
    "lama": "Qualidade — LaMa (IA, baixa um modelo)",
}
LAMA_FILE = "lama_fp32.onnx"
LAMA_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/" + LAMA_FILE
LAMA_SIZE = 512


# ------------------------------------------------------------------ máscara

def grow_mask(mask: np.ndarray, expandir: float) -> np.ndarray:
    """Engorda a máscara (0–100 ≈ até 1,5% do lado maior) para cobrir as bordas suaves da marca."""
    r = int(round(float(np.clip(expandir, 0, 100)) / 100.0 * 0.015 * max(mask.shape[:2])))
    binary = (mask > 127).astype(np.uint8) * 255
    if r < 1:
        return binary
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(binary, k)


def rect_mask(shape: tuple[int, ...], x: float, y: float, w: float, h: float) -> np.ndarray:
    """Máscara de um retângulo em coordenadas RELATIVAS (0–1)."""
    H, W = shape[:2]
    m = np.zeros((H, W), np.uint8)
    x0, y0 = int(round(x * W)), int(round(y * H))
    x1, y1 = int(round((x + w) * W)), int(round((y + h) * H))
    m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 255
    return m


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _composite(img: np.ndarray, filled: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cola o preenchido só dentro da máscara, com borda levemente suave."""
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), max(0.8, 0.0008 * max(mask.shape)))
    soft = np.maximum(soft, mask.astype(np.float32) / 255.0)[..., None]  # nunca deixa vazar a marca
    out = img.astype(np.float32) * (1.0 - soft) + filled.astype(np.float32) * soft
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ cv2.inpaint

def _inpaint_cv(rgb: np.ndarray, mask: np.ndarray, metodo: str, raio: float) -> np.ndarray:
    box = _bbox(mask)
    if box is None:
        return rgb.copy()
    H, W = rgb.shape[:2]
    r = max(2.0, float(np.clip(raio, 1, 100)) / 100.0 * 0.01 * max(H, W))
    pad = int(r * 3) + 4
    x0, y0, x1, y1 = max(0, box[0] - pad), max(0, box[1] - pad), min(W, box[2] + pad), min(H, box[3] + pad)
    roi, m = np.ascontiguousarray(rgb[y0:y1, x0:x1]), np.ascontiguousarray(mask[y0:y1, x0:x1])
    flag = cv2.INPAINT_TELEA if metodo == "telea" else cv2.INPAINT_NS
    filled = cv2.inpaint(roi, m, r, flag)
    out = rgb.copy()
    out[y0:y1, x0:x1] = filled
    return out


# ------------------------------------------------------------------------ LaMa

def lama_path() -> Path:
    return models_dir() / LAMA_FILE


def lama_available() -> bool:
    return lama_path().is_file()


def download_lama() -> Path:
    """Baixa o LaMa (~200 MB). Pode demorar: chame numa thread."""
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    dest = lama_path()
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(LAMA_URL, timeout=60, context=ctx) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if tmp.stat().st_size < 1_000_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Download do modelo LaMa incompleto.")
    tmp.replace(dest)
    return dest


_lama_session = None


def _lama():
    global _lama_session
    if _lama_session is None:
        if not lama_available():
            raise ModelNotDownloadedError(
                "O modelo LaMa ainda não foi baixado. Baixe-o na aba Marca d'água do programa."
            )
        _rembg()  # garante que o onnxruntime está instalado (mesma dependência)
        import onnxruntime as ort
        _lama_session = ort.InferenceSession(str(lama_path()), providers=["CPUExecutionProvider"])
    return _lama_session


def _lama_run(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Roda o LaMa em uma região (qualquer tamanho): reduz a 512×512 e devolve no tamanho original."""
    h, w = rgb.shape[:2]
    img512 = cv2.resize(rgb, (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_AREA)
    m512 = cv2.resize(mask, (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_NEAREST)
    x = (img512.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    m = (m512 > 127).astype(np.float32)[None, None]
    sess = _lama()
    feeds = {sess.get_inputs()[0].name: x, sess.get_inputs()[1].name: m}
    out = np.asarray(sess.run(None, feeds)[0])[0].transpose(1, 2, 0)
    if out.max() <= 2.0:  # modelo devolve 0–1 em vez de 0–255
        out = out * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)


WINDOW = LAMA_SIZE          # janela nativa do LaMa (px)
CTX = 96                    # contexto mínimo ao redor de cada pixel preenchido (px)
MAX_WINDOWS = 12            # acima disso, cai no método reduzido (mais rápido, mais mole)


def _inpaint_lama_downscaled(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Método reduzido: janela quadrada com contexto, escalada para 512 (qualquer tamanho de marca)."""
    box = _bbox(mask)
    if box is None:
        return rgb.copy()
    H, W = rgb.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    side = int(min(max(H, W), max(bw, bh) * 2.5))
    side = max(side, min(H, W, 256))
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    x0 = int(np.clip(cx - side // 2, 0, max(0, W - side)))
    y0 = int(np.clip(cy - side // 2, 0, max(0, H - side)))
    x1, y1 = min(W, x0 + side), min(H, y0 + side)
    roi, m = np.ascontiguousarray(rgb[y0:y1, x0:x1]), np.ascontiguousarray(mask[y0:y1, x0:x1])
    out = rgb.copy()
    out[y0:y1, x0:x1] = _lama_run(roi, m)
    return out


def _inpaint_lama(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """LaMa em janelas nativas de 512 px. Cada janela preenche só os pixels da máscara que têm
    contexto suficiente dentro dela; as seguintes já enxergam o que foi preenchido."""
    H, W = rgb.shape[:2]
    if min(H, W) < WINDOW:                       # foto menor que a janela: reduzida é o jeito
        return _inpaint_lama_downscaled(rgb, mask)
    out = rgb.copy()
    todo = mask.copy()
    for _ in range(MAX_WINDOWS):
        cols = np.flatnonzero(todo.any(axis=0))
        if len(cols) == 0:
            return out
        px = int(cols[0])                         # varre da esquerda para a direita, em faixas
        x0 = int(np.clip(px - CTX, 0, W - WINDOW))
        x1 = x0 + WINDOW
        ix0 = x0 + (CTX if x0 > 0 else 0)
        ix1 = x1 - (CTX if x1 < W else 0)
        rows = np.flatnonzero(todo[:, ix0:ix1].any(axis=1))   # linhas pendentes dentro da faixa
        y_min, y_max = int(rows[0]), int(rows[-1])
        span = y_max - y_min + 1
        if span <= WINDOW - 2 * CTX:              # a faixa cabe: centraliza nela
            y0 = y_min - (WINDOW - span) // 2
        else:
            y0 = y_min - CTX
        y0 = int(np.clip(y0, 0, H - WINDOW))
        y1 = y0 + WINDOW
        iy0 = y0 + (CTX if y0 > 0 else 0)
        iy1 = y1 - (CTX if y1 < H else 0)
        inner = np.zeros_like(todo)
        inner[iy0:iy1, ix0:ix1] = todo[iy0:iy1, ix0:ix1]
        if not inner.any():                       # segurança: nunca ficar preso
            inner[y_min:y_min + 1, px:px + 1] = todo[y_min:y_min + 1, px:px + 1]
        filled = _lama_run(np.ascontiguousarray(out[y0:y1, x0:x1]), np.ascontiguousarray(todo[y0:y1, x0:x1]))
        use = inner[y0:y1, x0:x1] > 0
        out[y0:y1, x0:x1][use] = filled[use]
        todo[inner > 0] = 0
    if todo.any():                                # marca enorme: termina com o método reduzido
        return _inpaint_lama_downscaled(out, todo)
    return out


# ------------------------------------------------------------------------ grão

def _noise_std(ch: np.ndarray, region: np.ndarray) -> float:
    """Desvio do 'grão' (alta frequência) de um canal dentro de uma região booleana."""
    hp = ch.astype(np.float32) - cv2.GaussianBlur(ch.astype(np.float32), (0, 0), 1.2)
    vals = hp[region]
    return float(vals.std()) if vals.size > 50 else 0.0


def restore_grain(original: np.ndarray, filled: np.ndarray, mask: np.ndarray, seed: int = 5) -> np.ndarray:
    """Recoloca no trecho preenchido o grão que a foto tem ao redor (o preenchimento sai liso).

    Mede o desvio do grão num anel em volta da máscara e no próprio trecho preenchido;
    acrescenta só o que falta. Determinístico (semente fixa).
    """
    H, W = mask.shape[:2]
    inside = mask > 0
    r = max(8, int(0.02 * max(H, W)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    ring = (cv2.dilate(mask, k) > 0) & ~inside
    gray_o = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    gray_f = cv2.cvtColor(filled, cv2.COLOR_RGB2GRAY)
    want, have = _noise_std(gray_o, ring), _noise_std(gray_f, inside)
    extra = float(np.sqrt(max(0.0, want * want - have * have)))
    if extra < 0.3:
        return filled
    noise = np.random.default_rng(seed).standard_normal((H, W)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.6)                   # blur suave deixa o grão mais natural
    hp = noise - cv2.GaussianBlur(noise, (0, 0), 1.2)
    noise *= extra / max(float(hp.std()), 1e-6)                    # calibra pela MESMA medida de grão
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), max(0.8, 0.0008 * max(H, W)))
    out = filled.astype(np.float32) + (noise * soft)[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------- público

def remove_with_mask(
    img: np.ndarray,
    mask: np.ndarray,
    metodo: str = "telea",
    expandir: float = 30,
    raio: float = 50,
    grao: bool = True,
) -> np.ndarray:
    """Apaga a região da máscara (uint8 HxW, >127 = apagar) e a preenche.
    `grao`: recoloca no trecho preenchido o grão da foto ao redor."""
    if metodo not in METHODS:
        raise ValueError(f"Método inválido: {metodo}")
    if mask.shape[:2] != img.shape[:2]:
        raise ValueError("A máscara deve ter o mesmo tamanho da imagem.")
    rgb, alpha = split_alpha(img)
    m = grow_mask(mask, expandir)
    if not m.any():
        return img.copy()
    rgb = np.ascontiguousarray(rgb)
    filled = _inpaint_lama(rgb, m) if metodo == "lama" else _inpaint_cv(rgb, m, metodo, raio)
    result = _composite(rgb, filled, m)
    if grao:
        result = restore_grain(rgb, result, m)
    return merge_alpha(result, alpha)


def remove_region(
    img: np.ndarray,
    x: float, y: float, w: float, h: float,
    metodo: str = "telea",
    expandir: float = 30,
    raio: float = 50,
    grao: bool = True,
) -> np.ndarray:
    """Apaga um retângulo em coordenadas RELATIVAS (0–1): reproduzível no lote."""
    return remove_with_mask(img, rect_mask(img.shape, x, y, w, h), metodo, expandir, raio, grao)
