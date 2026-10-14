"""Marca d'água em lote: modelos reutilizáveis.

Dois tipos de modelo, guardados em ~/.eggfotos/marcas:

1. MÁSCARA ANCORADA (`*.mask.png` + `.json`): a marcação feita à mão (pincel/clique/retângulo)
   recortada e ancorada ao canto/borda mais próximo, com tamanho e margens como fração da
   LARGURA da foto. Serve em fotos de tamanhos diferentes, desde que a marca escale com a
   largura. O preenchimento é o de inpaint.py (Telea/NS/LaMa).

2. ALPHA ESTIMADO (`*.alpha.png` + `.json`): para uma marca TRANSLÚCIDA branca ou preta que se
   repete na mesma posição em muitas fotos. Estima-se a opacidade α da marca olhando vários
   exemplos (um piso estatístico por pixel) e depois se inverte a composição
   J = α·cor + (1−α)·I  →  I = (J − α·cor) / (1 − α), recuperando os pixels (sem inventar textura).
   Exige fotos com a mesma proporção; funciona melhor com 20+ fotos variadas.

Use apenas em fotos suas ou com autorização de quem detém os direitos.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import inpaint
from .image_io import SUPPORTED_EXTENSIONS, load_image
from .imgutils import merge_alpha, split_alpha

KIND_MASK = "mask"
KIND_ALPHA = "alpha"
MIN_EXAMPLES = 8
ASPECT_TOLERANCE = 0.05   # diferença de proporção aceita entre modelo e foto
MAX_ALPHA = 0.95          # acima disso a inversão amplifica demais o ruído


def marks_dir() -> Path:
    """Pasta dos modelos. A variável EGGFOTOS_MARCAS (herdada pelos processos do lote) a substitui."""
    d = Path(os.environ.get("EGGFOTOS_MARCAS") or Path.home() / ".eggfotos" / "marcas")
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_name(nome: str) -> str:
    nome = re.sub(r'[\\/:*?"<>|\s]+', "_", nome.strip()).strip("._")
    if not nome:
        raise ValueError("Dê um nome ao modelo.")
    return nome


def _paths(nome: str, kind: str) -> tuple[Path, Path]:
    base = marks_dir() / f"{safe_name(nome)}.{kind}"
    return base.with_suffix(f".{kind}.png"), base.with_suffix(f".{kind}.json")


def list_templates(kind: str) -> list[str]:
    return sorted(p.name[: -len(f".{kind}.png")] for p in marks_dir().glob(f"*.{kind}.png"))


def template_exists(nome: str, kind: str) -> bool:
    png, js = _paths(nome, kind)
    return png.is_file() and js.is_file()


def delete_template(nome: str, kind: str) -> None:
    for p in _paths(nome, kind):
        p.unlink(missing_ok=True)


# ============================================================== máscara ancorada

@dataclass(frozen=True)
class MaskTemplate:
    crop: np.ndarray       # uint8 h×w (255 = apagar), recorte da caixa da marcação
    ax: str                # âncora horizontal: "l" | "c" | "r"
    ay: str                # âncora vertical:   "t" | "c" | "b"
    dx: float              # distância até a âncora, em frações da LARGURA da foto
    dy: float
    w: float               # tamanho da caixa, em frações da LARGURA da foto
    h: float


def make_mask_template(mask: np.ndarray) -> MaskTemplate:
    """Cria o modelo a partir de uma máscara (uint8 HxW) feita numa foto."""
    H, W = mask.shape[:2]
    ys, xs = np.nonzero(mask > 127)
    if len(xs) == 0:
        raise ValueError("A marcação está vazia.")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
    ax = "l" if cx < 1 / 3 else "r" if cx > 2 / 3 else "c"
    ay = "t" if cy < 1 / 3 else "b" if cy > 2 / 3 else "c"
    dx = {"l": x0, "r": W - x1, "c": (x0 + x1) / 2 - W / 2}[ax] / W
    dy = {"t": y0, "b": H - y1, "c": (y0 + y1) / 2 - H / 2}[ay] / W
    crop = np.where(mask[y0:y1, x0:x1] > 127, 255, 0).astype(np.uint8)
    return MaskTemplate(crop, ax, ay, dx, dy, (x1 - x0) / W, (y1 - y0) / W)


def place_mask(t: MaskTemplate, shape: tuple[int, ...]) -> np.ndarray:
    """Reposiciona o modelo numa foto de qualquer tamanho → máscara uint8 HxW."""
    H, W = shape[:2]
    bw, bh = max(1, round(t.w * W)), max(1, round(t.h * W))
    crop = cv2.resize(t.crop, (bw, bh), interpolation=cv2.INTER_LINEAR)
    x0 = {"l": t.dx * W, "r": W - t.dx * W - bw, "c": W / 2 + t.dx * W - bw / 2}[t.ax]
    y0 = {"t": t.dy * W, "b": H - t.dy * W - bh, "c": H / 2 + t.dy * W - bh / 2}[t.ay]
    x0, y0 = int(round(x0)), int(round(y0))
    out = np.zeros((H, W), np.uint8)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    ww, hh = min(bw - sx0, W - dx0), min(bh - sy0, H - dy0)
    if ww > 0 and hh > 0:
        out[dy0:dy0 + hh, dx0:dx0 + ww] = crop[sy0:sy0 + hh, sx0:sx0 + ww]
    return out


def save_mask_template(nome: str, t: MaskTemplate) -> Path:
    png, js = _paths(nome, KIND_MASK)
    cv2.imwrite(str(png), t.crop)
    js.write_text(json.dumps({"ax": t.ax, "ay": t.ay, "dx": t.dx, "dy": t.dy, "w": t.w, "h": t.h}),
                  encoding="utf-8")
    return png


def load_mask_template(nome: str) -> MaskTemplate:
    png, js = _paths(nome, KIND_MASK)
    if not (png.is_file() and js.is_file()):
        raise FileNotFoundError(f"Modelo de marca '{nome}' não encontrado em {marks_dir()}")
    crop = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
    return MaskTemplate(crop, **json.loads(js.read_text(encoding="utf-8")))


def remove_with_template(
    img: np.ndarray,
    nome: str,
    metodo: str = "telea",
    expandir: float = 30,
    raio: float = 50,
    grao: bool = True,
    contexto: float = 40,
    emenda: float = 20,
    grao_forca: float = 100,
) -> np.ndarray:
    """Remove a marca (modelo de máscara ancorada) de qualquer foto — usado pelo lote."""
    t = load_mask_template(nome)
    return inpaint.remove_with_mask(img, place_mask(t, img.shape), metodo, expandir, raio, grao,
                                    contexto, emenda, grao_forca)


# =============================================================== alpha estimado

@dataclass(frozen=True)
class AlphaTemplate:
    alpha: np.ndarray      # float32 HxW, 0..MAX_ALPHA
    cor: int               # 255 = marca branca, 0 = marca preta
    n: int = 0             # nº de fotos usadas no aprendizado


def _load_examples(folder: Path, max_side: int, limite: int) -> tuple[list[np.ndarray], int]:
    files = sorted(p for p in Path(folder).rglob("*")
                   if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file())
    if len(files) > limite:  # amostra espaçada: fotos variadas, não só as primeiras
        idx = np.linspace(0, len(files) - 1, limite).round().astype(int)
        files = [files[i] for i in idx]
    ref_size: tuple[int, int] | None = None
    imgs, skipped = [], 0
    for f in files:
        try:
            img, _ = load_image(f)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        rgb, _ = split_alpha(img)
        H, W = rgb.shape[:2]
        if ref_size is None:
            s = min(1.0, max_side / max(H, W))
            ref_size = (max(1, round(W * s)), max(1, round(H * s)))
        if abs((W / H) / (ref_size[0] / ref_size[1]) - 1) > ASPECT_TOLERANCE:
            skipped += 1
            continue
        imgs.append(cv2.resize(np.ascontiguousarray(rgb), ref_size, interpolation=cv2.INTER_AREA))
    return imgs, skipped


def estimate_alpha(images: list[np.ndarray], cor: int = 255, percentil: float = 3.0,
                   traco_max: float = 0.04) -> AlphaTemplate:
    """Estima a opacidade da marca a partir de fotos do mesmo tamanho (uint8 RGB).

    Para marca branca, em cada pixel o valor mais escuro entre as fotos é
    P ≈ α·255 + (1−α)·F, com F o "piso" natural da cena naquele ponto (varia devagar).
    F é estimado por uma abertura morfológica de P (remove traços mais finos que `traco_max`
    × lado maior, ou seja, a marca) suavizada. O viés do piso é medido no fundo da própria
    imagem (a marca ocupa pouco) e descontado; o que sobra acima do ruído é α·(255−F).
    Marca preta: o mesmo com as imagens invertidas.
    """
    if len(images) < MIN_EXAMPLES:
        raise ValueError(f"Preciso de pelo menos {MIN_EXAMPLES} fotos com a mesma proporção "
                         f"(encontrei {len(images)}). Quanto mais variadas, melhor.")
    stack = np.stack(images)
    if cor == 0:
        stack = 255 - stack
    k = min(len(stack) - 1, max(0, int(round(percentil / 100.0 * len(stack)))))
    P = np.partition(stack, k, axis=0)[k].astype(np.float32)       # k-ésimo menor valor por pixel
    big = max(P.shape[:2])
    P = cv2.GaussianBlur(P, (0, 0), max(0.8, 0.0012 * big))
    ks = int(round(traco_max * big)) * 2 + 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    F = cv2.GaussianBlur(cv2.morphologyEx(P, cv2.MORPH_OPEN, kern), (0, 0), ks / 3.0)
    d = (P - F).mean(axis=2)
    d -= np.median(d)                                              # viés do piso (fundo domina)
    noise = 1.4826 * np.median(np.abs(d))
    d[d < max(2.0, 3.0 * noise)] = 0.0
    a = np.clip(d / np.maximum(255.0 - F.mean(axis=2), 1.0), 0.0, MAX_ALPHA)
    return AlphaTemplate(a.astype(np.float32), int(cor), len(images))


def learn_from_folder(folder: str | Path, nome: str, cor: int = 255, max_side: int = 1200,
                      limite: int = 60) -> tuple[AlphaTemplate, int]:
    """Lê as fotos da pasta, estima e SALVA o modelo. Devolve (modelo, fotos ignoradas)."""
    imgs, skipped = _load_examples(Path(folder), max_side, limite)
    t = estimate_alpha(imgs, cor)
    if not (t.alpha > 0).any():
        raise ValueError("Não encontrei uma marca repetida nessas fotos. "
                         "Confira a cor (branca/preta) e se estão na mesma posição.")
    save_alpha_template(nome, t)
    return t, skipped


def save_alpha_template(nome: str, t: AlphaTemplate) -> Path:
    png, js = _paths(nome, KIND_ALPHA)
    cv2.imwrite(str(png), np.round(t.alpha * 65535).astype(np.uint16))
    js.write_text(json.dumps({"cor": t.cor, "n": t.n}), encoding="utf-8")
    return png


def load_alpha_template(nome: str) -> AlphaTemplate:
    png, js = _paths(nome, KIND_ALPHA)
    if not (png.is_file() and js.is_file()):
        raise FileNotFoundError(f"Modelo de marca '{nome}' não encontrado em {marks_dir()}")
    a = cv2.imread(str(png), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
    meta = json.loads(js.read_text(encoding="utf-8"))
    return AlphaTemplate(a, int(meta["cor"]), int(meta.get("n", 0)))


def remove_alpha(img: np.ndarray, t: AlphaTemplate, forca: float = 100) -> np.ndarray:
    """Inverte a composição da marca: I = (J − α·cor)/(1 − α)."""
    rgb, alpha_ch = split_alpha(img)
    H, W = rgb.shape[:2]
    h0, w0 = t.alpha.shape
    if abs((W / H) / (w0 / h0) - 1) > ASPECT_TOLERANCE:
        raise ValueError(f"A foto ({W}×{H}) tem proporção diferente das fotos usadas para "
                         f"aprender a marca ({w0}×{h0}).")
    a = cv2.resize(t.alpha, (W, H), interpolation=cv2.INTER_LINEAR)
    a = np.clip(a * float(np.clip(forca, 0, 150)) / 100.0, 0.0, MAX_ALPHA)[..., None]
    j = rgb.astype(np.float32)
    out = (j - a * t.cor) / (1.0 - a)
    return merge_alpha(np.clip(out + 0.5, 0, 255).astype(np.uint8), alpha_ch)


def remove_estimated(img: np.ndarray, nome: str, forca: float = 100) -> np.ndarray:
    """Remove a marca translúcida aprendida (modelo `nome`) — usado pelo lote."""
    return remove_alpha(img, load_alpha_template(nome), forca)
