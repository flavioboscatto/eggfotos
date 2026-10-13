"""Remoção de objetos / marca d'água por preenchimento (inpainting).

Dois métodos, mesma interface (imagem + máscara uint8, 255 = apagar):
  - "telea" / "ns": cv2.inpaint, rápido e sem modelo; bom para marcas finas e fundos lisos.
  - "lama": rede LaMa (ONNX), bem melhor em texturas e áreas grandes; modelo baixado sob demanda.

Só a região da máscara (mais contexto) é processada, então é rápido mesmo em fotos grandes.
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


def _inpaint_lama(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    if box is None:
        return rgb.copy()
    H, W = rgb.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    # janela quadrada com bastante contexto ao redor da marca
    side = int(min(max(H, W), max(bw, bh) * 2.5, max(H, W)))
    side = max(side, min(H, W, 256))
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    x0 = int(np.clip(cx - side // 2, 0, max(0, W - side)))
    y0 = int(np.clip(cy - side // 2, 0, max(0, H - side)))
    x1, y1 = min(W, x0 + side), min(H, y0 + side)
    roi, m = np.ascontiguousarray(rgb[y0:y1, x0:x1]), np.ascontiguousarray(mask[y0:y1, x0:x1])
    filled = _lama_run(roi, m)
    out = rgb.copy()
    out[y0:y1, x0:x1] = filled
    return out


# --------------------------------------------------------------------- público

def remove_with_mask(
    img: np.ndarray,
    mask: np.ndarray,
    metodo: str = "telea",
    expandir: float = 30,
    raio: float = 50,
) -> np.ndarray:
    """Apaga a região da máscara (uint8 HxW, >127 = apagar) e a preenche."""
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
    return merge_alpha(_composite(rgb, filled, m), alpha)


def remove_region(
    img: np.ndarray,
    x: float, y: float, w: float, h: float,
    metodo: str = "telea",
    expandir: float = 30,
    raio: float = 50,
) -> np.ndarray:
    """Apaga um retângulo em coordenadas RELATIVAS (0–1): reproduzível no lote."""
    return remove_with_mask(img, rect_mask(img.shape, x, y, w, h), metodo, expandir, raio)
