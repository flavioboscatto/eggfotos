"""Seleção por cliques com MobileSAM (ONNX): clique no objeto → máscara.

Duas etapas, como em background.py:
  1. embed(img)                  -> Embedding   (~0,5 s; cacheável por foto)
  2. predict(emb, pontos)        -> máscara     (instantâneo; refina a cada clique)

Pontos são RELATIVOS (0–1) e têm rótulo: 1 = "é objeto", 0 = "não é objeto".
Os dois modelos (~45 MB) ficam em ~/.eggfotos/modelos e são baixados sob demanda.
"""
from __future__ import annotations

import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .background import ModelNotDownloadedError, _rembg, models_dir
from .imgutils import split_alpha

BASE_URL = "https://huggingface.co/Acly/MobileSAM/resolve/main/"
ENCODER_FILE = "mobile_sam_image_encoder.onnx"
DECODER_FILE = "sam_mask_decoder_single.onnx"
MODEL_FILES = (ENCODER_FILE, DECODER_FILE)
SAM_SIZE = 1024  # o encoder espera a foto com o lado maior = 1024 (ele não redimensiona sozinho)

Point = tuple[float, float, int]  # x, y relativos e rótulo (1 objeto / 0 fundo)


@dataclass
class Embedding:
    data: np.ndarray        # (1, 256, 64, 64)
    scale: float            # fator foto → imagem 1024
    size: tuple[int, int]   # (altura, largura) da foto original


def model_paths() -> tuple[Path, Path]:
    d = models_dir()
    return d / ENCODER_FILE, d / DECODER_FILE


def model_available() -> bool:
    return all(p.is_file() for p in model_paths())


def download_models() -> None:
    """Baixa encoder e decoder (~45 MB). Pode demorar: chame numa thread."""
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    for name, dest in zip(MODEL_FILES, model_paths()):
        if dest.is_file():
            continue
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(BASE_URL + name, timeout=60, context=ctx) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        if tmp.stat().st_size < 1_000_000:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"O download de {name} não terminou. Tente de novo.")
        tmp.replace(dest)


_sessions: dict[str, object] = {}


def _session(kind: str):
    if kind not in _sessions:
        if not model_available():
            raise ModelNotDownloadedError(
                "Os modelos de seleção por clique ainda não foram baixados. "
                "Baixe-os pelo botão da aba."
            )
        _rembg()  # confere se o onnxruntime está instalado
        import onnxruntime as ort
        path = model_paths()[0 if kind == "encoder" else 1]
        _sessions[kind] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return _sessions[kind]


def embed(img: np.ndarray) -> Embedding:
    """Calcula o embedding da foto (a parte cara)."""
    rgb, _ = split_alpha(img)
    H, W = rgb.shape[:2]
    s = SAM_SIZE / max(H, W)
    small = cv2.resize(np.ascontiguousarray(rgb), (max(1, round(W * s)), max(1, round(H * s))),
                       interpolation=cv2.INTER_AREA)
    data = _session("encoder").run(None, {"input_image": small.astype(np.float32)})[0]
    return Embedding(data, s, (H, W))


def predict(emb: Embedding, pontos: list[Point]) -> np.ndarray:
    """Máscara uint8 HxW (0/255) para os pontos dados. Sem pontos, devolve máscara vazia."""
    H, W = emb.size
    if not pontos:
        return np.zeros((H, W), np.uint8)
    coords = [[x * W * emb.scale, y * H * emb.scale] for x, y, _ in pontos]
    labels = [float(l) for _, _, l in pontos]
    coords.append([0.0, 0.0])   # ponto de preenchimento exigido pelo SAM quando não há caixa
    labels.append(-1.0)
    out = _session("decoder").run(None, {
        "image_embeddings": emb.data,
        "point_coords": np.array([coords], np.float32),
        "point_labels": np.array([labels], np.float32),
        "mask_input": np.zeros((1, 1, 256, 256), np.float32),
        "has_mask_input": np.zeros(1, np.float32),
        "orig_im_size": np.array([H, W], np.float32),
    })
    mask = np.asarray(out[0])[0, 0] > 0
    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return mask.astype(np.uint8) * 255


def select_object(img: np.ndarray, pontos: list[Point]) -> np.ndarray:
    """Atalho (sem cache): embedding + máscara."""
    return predict(embed(img), pontos)
