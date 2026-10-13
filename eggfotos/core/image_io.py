"""Leitura e gravação de imagens com Pillow, preservando EXIF e perfil de cor."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, ImageOps

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
_FORMAT_BY_EXT = {
    ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP",
    ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF",
}
_METADATA_FORMATS = {"JPEG", "PNG", "WEBP", "TIFF"}
_ORIENTATION_TAG = 0x0112

OPEN_FILTER = "Imagens (" + " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS) + ")"


def is_supported(path: str | os.PathLike) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def load_image(path: str | os.PathLike) -> tuple[np.ndarray, dict]:
    """Abre a imagem já girada conforme o EXIF.

    Retorna (array RGB/RGBA uint8, metadados). O EXIF devolvido tem a
    orientação zerada, pois a rotação já foi aplicada aos pixels.
    """
    path = Path(path)
    with Image.open(path) as im:
        im.load()
        fmt = im.format
        icc = im.info.get("icc_profile")
        exif = im.getexif()
        im = ImageOps.exif_transpose(im)
        has_alpha = im.mode in ("RGBA", "LA", "PA") or (
            im.mode == "P" and "transparency" in im.info
        )
        im = im.convert("RGBA" if has_alpha else "RGB")
        arr = np.array(im, dtype=np.uint8)

    exif_bytes = None
    if exif and len(exif):
        if _ORIENTATION_TAG in exif:
            exif[_ORIENTATION_TAG] = 1
        try:
            exif_bytes = exif.tobytes()
        except Exception:  # EXIF corrompido: ignora em vez de falhar
            exif_bytes = None
    meta = {"path": str(path), "format": fmt, "exif": exif_bytes, "icc_profile": icc}
    return arr, meta


def flatten_alpha(img: np.ndarray, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Compõe uma imagem RGBA sobre uma cor sólida, retornando RGB."""
    if img.ndim != 3 or img.shape[2] != 4:
        return img
    a = img[..., 3:4].astype(np.float32) / 255.0
    bg = np.array(color, dtype=np.float32)
    out = img[..., :3].astype(np.float32) * a + bg * (1.0 - a)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def save_image(
    arr: np.ndarray,
    path: str | os.PathLike,
    quality: int = 92,
    meta: dict | None = None,
    keep_metadata: bool = True,
) -> Path:
    """Grava a imagem. O formato vem da extensão do arquivo.

    A gravação é atômica (arquivo temporário + rename), para nunca deixar
    um arquivo corrompido se algo falhar no meio.
    """
    path = Path(path)
    fmt = _FORMAT_BY_EXT.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(f"Formato não suportado para gravação: {path.suffix}")

    arr = np.ascontiguousarray(arr)
    if fmt in ("JPEG", "BMP") and arr.ndim == 3 and arr.shape[2] == 4:
        arr = flatten_alpha(arr)
    if arr.ndim == 2:
        im = Image.fromarray(arr, "L")
    elif arr.shape[2] == 4:
        im = Image.fromarray(arr, "RGBA")
    else:
        im = Image.fromarray(arr, "RGB")

    kwargs: dict = {}
    q = int(max(1, min(100, quality)))
    if fmt == "JPEG":
        kwargs.update(quality=q, optimize=True, subsampling=0 if q >= 90 else 2)
        # optimize=True precisa de buffer do tamanho da imagem; sem isto, fotos com
        # muito detalhe falham com "Suspension not allowed here" (limitação do libjpeg).
        ImageFile.MAXBLOCK = max(ImageFile.MAXBLOCK, im.size[0] * im.size[1] * 4)
    elif fmt == "WEBP":
        kwargs.update(quality=q, method=4)
    elif fmt == "PNG":
        kwargs.update(compress_level=6)
    if keep_metadata and meta and fmt in _METADATA_FORMATS:
        if meta.get("exif"):
            kwargs["exif"] = meta["exif"]
        if meta.get("icc_profile"):
            kwargs["icc_profile"] = meta["icc_profile"]

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        im.save(tmp, format=fmt, **kwargs)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return path
