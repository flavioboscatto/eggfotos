"""Remoção de fundo: máscara por rede neural (rembg/ONNX), refino da borda,
descontaminação de cor e troca do fundo.

Fluxo em duas etapas, para a interface poder ajustar sem rodar o modelo de novo:
  1. compute_matte(img, modelo)            -> máscara alpha uint8 HxW (cara, cacheável)
  2. apply_matte(img, matte, ...)          -> imagem final (rápida, ajustável ao vivo)
remove_background() junta as duas e é o que o lote usa.

Os modelos ficam em ~/.eggfotos/modelos (nunca dentro do projeto) e são baixados
só quando o usuário pede. rembg/onnxruntime são importados sob demanda.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .blur import _hex_to_rgb
from .imgutils import merge_alpha, rel_sigma, split_alpha

# chave -> (nome no rembg, rótulo)
MODELS: dict[str, tuple[str, str]] = {
    "rapido": ("u2netp", "Rápido (leve, ~4 MB)"),
    "equilibrado": ("isnet-general-use", "Equilibrado (~170 MB)"),
    "preciso": ("birefnet-general-lite", "Preciso — BiRefNet lite (~170 MB)"),
}
DEFAULT_MODEL = "equilibrado"

BACKGROUND_TRANSPARENT = "transparente"


class BackgroundDependencyError(RuntimeError):
    """rembg/onnxruntime não instalados."""


class ModelNotDownloadedError(RuntimeError):
    """O modelo ainda não foi baixado."""


def models_dir() -> Path:
    d = Path.home() / ".eggfotos" / "modelos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_path(modelo: str) -> Path:
    """Onde está o modelo. O rembg 2.x guarda em <pasta>/models/<nome>/<nome>.onnx;
    versões antigas, em <pasta>/<nome>.onnx. Aceita os dois."""
    nome = MODELS[modelo][0]
    candidatos = [
        models_dir() / "models" / nome / f"{nome}.onnx",
        models_dir() / f"{nome}.onnx",
    ]
    return next((c for c in candidatos if c.is_file()), candidatos[0])


def model_available(modelo: str) -> bool:
    return model_path(modelo).is_file()


def _rembg():
    os.environ.setdefault("U2NET_HOME", str(models_dir()))
    try:
        import rembg
    except ImportError as e:
        raise BackgroundDependencyError(
            "Remoção de fundo precisa de rembg e onnxruntime:\n"
            "pip install -r requirements.txt"
        ) from e
    return rembg


def download_model(modelo: str) -> Path:
    """Baixa o modelo (se faltar) para a pasta de modelos. Pode demorar: chame numa thread."""
    if modelo not in MODELS:
        raise ValueError(f"Modelo desconhecido: {modelo}")
    rembg = _rembg()
    rembg.new_session(MODELS[modelo][0])  # baixa e valida o arquivo
    return model_path(modelo)


_sessions: dict[str, object] = {}


def _session(modelo: str):
    if modelo not in MODELS:
        raise ValueError(f"Modelo desconhecido: {modelo}")
    if modelo not in _sessions:
        if not model_available(modelo):
            raise ModelNotDownloadedError(
                f"O modelo '{MODELS[modelo][1]}' ainda não foi baixado. "
                "Baixe-o na aba Fundo do programa."
            )
        _sessions[modelo] = _rembg().new_session(MODELS[modelo][0])
    return _sessions[modelo]


# ------------------------------------------------------------------- máscara

def compute_matte(img: np.ndarray, modelo: str = DEFAULT_MODEL) -> np.ndarray:
    """Máscara alpha (uint8 HxW, 255 = objeto) estimada pelo modelo."""
    rgb, _ = split_alpha(img)
    session = _session(modelo)
    matte = _rembg().remove(np.ascontiguousarray(rgb), session=session, only_mask=True)
    matte = np.asarray(matte)
    if matte.ndim == 3:
        matte = matte[..., 0]
    return np.ascontiguousarray(matte.astype(np.uint8))


def refine_alpha(alpha: np.ndarray, encolher: float = 0, suavizar: float = 0) -> np.ndarray:
    """Refina a borda. `encolher` (−100..100, % do lado maior × 0,5) afina (+) ou engorda (−);
    `suavizar` (0..100) desfoca a borda. Valores relativos ao tamanho da foto."""
    out = alpha
    ref = max(alpha.shape[:2])
    if encolher:
        r = int(round(abs(encolher) / 100.0 * 0.005 * ref))
        if r >= 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            out = cv2.erode(out, k) if encolher > 0 else cv2.dilate(out, k)
    if suavizar > 0:
        out = cv2.GaussianBlur(out, (0, 0), max(0.3, suavizar / 100.0 * 0.004 * ref))
    return out


def decontaminate(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Remove o halo da cor do fundo antigo nas bordas semitransparentes.

    Estima a cor do objeto a partir do interior sólido (convolução normalizada) e a
    usa onde o alpha é parcial, preservando o interior intacto.
    """
    a = alpha.astype(np.float32) / 255.0
    core = (a > 0.97).astype(np.float32)
    sigma = rel_sigma(rgb, 0.004, 1.0)
    num = cv2.GaussianBlur(rgb.astype(np.float32) * core[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(core, (0, 0), sigma)[..., None]
    est = num / np.maximum(den, 1e-4)
    # onde não há interior por perto, mantém a cor original
    use = np.clip((1.0 - a), 0.0, 1.0)[..., None] * (den > 1e-3)
    mixed = rgb.astype(np.float32) * (1.0 - use) + est * use
    return np.clip(mixed + 0.5, 0, 255).astype(np.uint8)


def apply_matte(
    img: np.ndarray,
    matte: np.ndarray,
    encolher: float = 0,
    suavizar: float = 20,
    descontaminar: bool = True,
    fundo: str = BACKGROUND_TRANSPARENT,
    cor: str = "#ffffff",
) -> np.ndarray:
    """Aplica a máscara: RGBA se fundo="transparente", RGB com a cor sólida `cor` se fundo="cor"."""
    rgb, old_alpha = split_alpha(img)
    if matte.shape[:2] != rgb.shape[:2]:
        matte = cv2.resize(matte, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    alpha = refine_alpha(matte, encolher, suavizar)
    if old_alpha is not None:
        alpha = (alpha.astype(np.uint16) * old_alpha[..., 0] // 255).astype(np.uint8)
    fg = decontaminate(rgb, alpha) if descontaminar else np.ascontiguousarray(rgb)

    if fundo == BACKGROUND_TRANSPARENT:
        return merge_alpha(fg, alpha[..., None])
    if fundo != "cor":
        raise ValueError(f"Fundo inválido: {fundo}")
    bg = np.empty_like(fg)
    bg[...] = _hex_to_rgb(cor)
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    out = fg.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def remove_background(
    img: np.ndarray,
    modelo: str = DEFAULT_MODEL,
    encolher: float = 0,
    suavizar: float = 20,
    descontaminar: bool = True,
    fundo: str = BACKGROUND_TRANSPARENT,
    cor: str = "#ffffff",
) -> np.ndarray:
    """Remove o fundo de ponta a ponta (usado pelo lote)."""
    return apply_matte(img, compute_matte(img, modelo), encolher, suavizar, descontaminar, fundo, cor)


def blur_background(
    img: np.ndarray,
    matte: np.ndarray,
    intensidade: float = 50,
    encolher: float = 0,
    suavizar: float = 20,
) -> np.ndarray:
    """Desfoca tudo que NÃO é o objeto (efeito "retrato"). Mantém o alpha da foto."""
    rgb, old_alpha = split_alpha(img)
    if matte.shape[:2] != rgb.shape[:2]:
        matte = cv2.resize(matte, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    alpha = refine_alpha(matte, encolher, suavizar).astype(np.float32)[..., None] / 255.0
    t = float(np.clip(intensidade, 1, 100)) / 100.0
    sigma = max(0.5, t * 0.03 * max(rgb.shape[:2]))
    blurred = cv2.GaussianBlur(np.ascontiguousarray(rgb), (0, 0), sigma)
    out = rgb.astype(np.float32) * alpha + blurred.astype(np.float32) * (1.0 - alpha)
    return merge_alpha(np.clip(out + 0.5, 0, 255).astype(np.uint8), old_alpha)


def blur_background_auto(
    img: np.ndarray,
    modelo: str = DEFAULT_MODEL,
    intensidade: float = 50,
    encolher: float = 0,
    suavizar: float = 20,
) -> np.ndarray:
    """Desfoque de fundo de ponta a ponta (usado pelo lote)."""
    return blur_background(img, compute_matte(img, modelo), intensidade, encolher, suavizar)
