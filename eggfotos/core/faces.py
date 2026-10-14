"""Detecção de rostos (YuNet, via OpenCV) e desfoque automático.

Mesmo princípio de background.py: detecção (cara, cacheável) separada do desfoque
(rápido). O modelo (~230 KB) fica em ~/.eggfotos/modelos e é baixado só sob demanda.
Caixas são RELATIVAS (0–1) para servirem em qualquer resolução.
"""
from __future__ import annotations

import ssl
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .background import ModelNotDownloadedError, models_dir
from .blur import BLUR_TYPES, blur_with_mask
from .imgutils import split_alpha

MODEL_FILE = "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + MODEL_FILE
)
DETECT_MAX_SIDE = 1280   # a detecção roda numa cópia reduzida (rápido e igual na prévia e no final)
MIN_SCORE = 0.3          # piso da detecção; o filtro fino é a "sensibilidade"

Box = tuple[float, float, float, float, float]  # x, y, w, h (relativos) e score


def model_path() -> Path:
    return models_dir() / MODEL_FILE


def model_available() -> bool:
    return model_path().is_file()


def download_model() -> Path:
    """Baixa o YuNet. Pode demorar: chame numa thread."""
    dest = model_path()
    tmp = dest.with_suffix(".part")
    try:  # o Python do macOS não traz certificados raiz; usa os do certifi
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(MODEL_URL, timeout=60, context=ctx) as r, open(tmp, "wb") as f:
        f.write(r.read())
    if tmp.stat().st_size < 10_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("O download do modelo de rostos não terminou. Tente de novo.")
    tmp.replace(dest)
    return dest


def score_threshold(sensibilidade: float) -> float:
    """Sensibilidade 1–100 → limiar de confiança (mais sensível = aceita mais rostos)."""
    return 0.9 - 0.6 * float(np.clip(sensibilidade, 0, 100)) / 100.0


def detect_faces(img: np.ndarray) -> list[Box]:
    """Rostos encontrados: lista de (x, y, w, h, score), coordenadas relativas."""
    if not model_available():
        raise ModelNotDownloadedError(
            "O modelo de rostos ainda não foi baixado. Baixe-o na aba Ocultar rosto do programa."
        )
    rgb, _ = split_alpha(img)
    H, W = rgb.shape[:2]
    s = min(1.0, DETECT_MAX_SIDE / max(H, W))
    small = cv2.resize(rgb, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    det = cv2.FaceDetectorYN.create(str(model_path()), "", (w, h), MIN_SCORE, 0.3, 5000)
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    out: list[Box] = []
    if faces is None:
        return out
    for f in faces:
        x, y, bw, bh, score = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14])
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(w), x + bw), min(float(h), y + bh)
        if x1 > x0 and y1 > y0:
            out.append((x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h, score))
    return out


def faces_mask(shape: tuple[int, ...], boxes: list[Box], margem: float = 30) -> np.ndarray:
    """Máscara de elipses sobre as caixas, ampliadas em `margem` % para cobrir testa/queixo."""
    H, W = shape[:2]
    mask = np.zeros((H, W), np.uint8)
    k = float(np.clip(margem, 0, 200)) / 100.0
    for x, y, w, h, _ in boxes:
        cx, cy = (x + w / 2) * W, (y + h / 2) * H
        rx, ry = w * W / 2 * (1 + k), h * H / 2 * (1 + k)
        cv2.ellipse(mask, (round(cx), round(cy)), (max(1, round(rx)), max(1, round(ry))),
                    0, 0, 360, 255, -1)
    return mask


def blur_boxes(
    img: np.ndarray,
    boxes: list[Box],
    sensibilidade: float = 60,
    margem: float = 30,
    tipo: str = "mosaico",
    intensidade: float = 70,
    cor: str = "#000000",
) -> np.ndarray:
    """Desfoca os rostos já detectados cujo score passa no limiar da sensibilidade."""
    if tipo not in BLUR_TYPES:
        raise ValueError(f"Tipo de desfoque inválido: {tipo}")
    thr = score_threshold(sensibilidade)
    chosen = [b for b in boxes if b[4] >= thr]
    if not chosen:
        return img.copy()
    return blur_with_mask(img, faces_mask(img.shape, chosen, margem), tipo, intensidade, cor)


def blur_faces(
    img: np.ndarray,
    sensibilidade: float = 60,
    margem: float = 30,
    tipo: str = "mosaico",
    intensidade: float = 70,
    cor: str = "#000000",
) -> np.ndarray:
    """Detecta e desfoca rostos de ponta a ponta (usado pelo lote)."""
    return blur_boxes(img, detect_faces(img), sensibilidade, margem, tipo, intensidade, cor)
