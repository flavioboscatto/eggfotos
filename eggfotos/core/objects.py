"""Reconhecimento de objetos por tipo (pessoas, carros, prédios, casas, barcos…) com SegFormer (ONNX).

O SegFormer entrega, para cada pixel, uma de 150 classes (ADE20K). Aqui elas são agrupadas em
CATEGORIAS em português. O usuário escolhe, por categoria, o que MANTER e o que APAGAR na aba Fundo.

Duas etapas, como em background.py:
  1. scan(img, modelo)                       -> ObjectMap   (cara, cacheável por foto)
  2. select(img, manter, apagar, ...)        -> máscara alpha (uint8 HxW, 255 = fica na foto)

As bordas da máscara por categoria são grosseiras (a rede trabalha em 1/4 da resolução). Por isso,
se o MobileSAM estiver disponível, cada objeto é refinado com um clique "invisível" no seu centro
(refine_with_sam). Sem o MobileSAM, usa-se a máscara da rede, suavizada.

O modelo fica em ~/.eggfotos/modelos e é baixado só quando o usuário pede.
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

BASE_URL = "https://huggingface.co/Xenova/"

# chave -> (arquivo local, caminho no Hugging Face, rótulo)
MODELS: dict[str, tuple[str, str, str]] = {
    "rapido": ("segformer_b0_ade.onnx", "segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model.onnx",
               "Rápido — SegFormer B0 (~15 MB)"),
    "preciso": ("segformer_b2_ade.onnx", "segformer-b2-finetuned-ade-512-512/resolve/main/onnx/model.onnx",
                "Preciso — SegFormer B2 (~110 MB)"),
}
DEFAULT_MODEL = "rapido"
SCAN_SIDE = 768          # lado maior da entrada da rede (múltiplo de 32)
MIN_FRACTION = 0.002     # categoria com menos que isso da foto não é listada
MIN_PIECE = 0.0005       # objeto menor que isso (fração da foto) é ignorado ao refinar


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    ids: tuple[int, ...]    # classes ADE20K


CATEGORIES: dict[str, Category] = {c.key: c for c in (
    Category("pessoas", "Pessoas", (12,)),
    Category("veiculos", "Carros e veículos", (20, 80, 83, 102, 116, 127)),   # car bus truck van minibike bicycle
    Category("predios", "Prédios e casas", (1, 25, 48, 84, 79)),               # building house skyscraper tower hovel
    Category("barcos", "Barcos e navios", (76, 103)),                          # boat ship
    Category("animais", "Animais", (126,)),
    Category("avioes", "Aviões", (90,)),
    Category("arvores", "Árvores e plantas", (4, 17, 72, 66)),                 # tree plant palm flower
    Category("ceu", "Céu", (2,)),
    Category("agua", "Água (mar, rio, lago)", (21, 26, 60, 128, 113)),
    Category("montanhas", "Montanhas e pedras", (16, 68, 34)),
    Category("estradas", "Ruas e calçadas", (6, 11, 52)),
)}
# Coisas contáveis, de contorno bem definido: o refino por SAM ajuda. Já céu, árvores, água, montanhas e
# ruas são "massas" sem contorno de objeto: o SAM só atrapalha, então ficam com a máscara da rede.
INSTANCE_KEYS = ("pessoas", "veiculos", "barcos", "animais", "avioes")
_ORDER = list(CATEGORIES)
_CLASS_TO_CAT = np.zeros(256, np.uint8)       # classe ADE → índice da categoria + 1 (0 = nenhuma)
for _i, _k in enumerate(_ORDER):
    for _c in CATEGORIES[_k].ids:
        _CLASS_TO_CAT[_c] = _i + 1


def category_keys() -> list[str]:
    return list(CATEGORIES)


def category_label(key: str) -> str:
    return CATEGORIES[key].label


def parse_keys(texto: str) -> list[str]:
    """'pessoas, barcos' → ['pessoas', 'barcos'] (valida as chaves)."""
    keys = [k.strip().lower() for k in str(texto or "").replace(";", ",").split(",") if k.strip()]
    bad = [k for k in keys if k not in CATEGORIES]
    if bad:
        raise ValueError(f"Objeto desconhecido: {', '.join(bad)}. Opções: {', '.join(CATEGORIES)}")
    return list(dict.fromkeys(keys))


# -------------------------------------------------------------------- modelo

def model_path(modelo: str) -> Path:
    return models_dir() / MODELS[modelo][0]


def model_available(modelo: str = DEFAULT_MODEL) -> bool:
    return model_path(modelo).is_file()


def download_model(modelo: str = DEFAULT_MODEL) -> Path:
    """Baixa o modelo de objetos. Pode demorar: chame numa thread."""
    if modelo not in MODELS:
        raise ValueError(f"Modelo desconhecido: {modelo}")
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    dest = model_path(modelo)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(BASE_URL + MODELS[modelo][1], timeout=60, context=ctx) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if tmp.stat().st_size < 1_000_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("O download do modelo de objetos não terminou. Tente de novo.")
    tmp.replace(dest)
    return dest


_sessions: dict[str, object] = {}


def _session(modelo: str):
    if modelo not in MODELS:
        raise ValueError(f"Modelo desconhecido: {modelo}")
    if modelo not in _sessions:
        if not model_available(modelo):
            raise ModelNotDownloadedError(
                f"O modelo '{MODELS[modelo][2]}' ainda não foi baixado. Baixe-o na aba Fundo.")
        _rembg()  # confere se o onnxruntime está instalado
        import onnxruntime as ort
        _sessions[modelo] = ort.InferenceSession(str(model_path(modelo)), providers=["CPUExecutionProvider"])
    return _sessions[modelo]


# ---------------------------------------------------------------------- scan

@dataclass
class ObjectMap:
    labels: np.ndarray          # (h, w) uint8: índice da categoria + 1 (0 = nenhuma), na resolução da rede
    size: tuple[int, int]       # (altura, largura) da foto original


_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _scan_size(H: int, W: int) -> tuple[int, int]:
    s = SCAN_SIDE / max(H, W)
    h = max(32, int(round(H * s / 32)) * 32)
    w = max(32, int(round(W * s / 32)) * 32)
    return h, w


def scan(img: np.ndarray, modelo: str = DEFAULT_MODEL) -> ObjectMap:
    """Classifica cada pixel (a parte cara). Determinístico."""
    rgb, _ = split_alpha(img)
    H, W = rgb.shape[:2]
    h, w = _scan_size(H, W)
    small = cv2.resize(np.ascontiguousarray(rgb), (w, h), interpolation=cv2.INTER_AREA)
    x = ((small.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]
    logits = np.asarray(_session(modelo).run(None, {"pixel_values": np.ascontiguousarray(x)})[0])[0]
    classes = logits.argmax(axis=0).astype(np.uint8)
    return ObjectMap(_CLASS_TO_CAT[classes], (H, W))


def category_mask(om: ObjectMap, keys: list[str]) -> np.ndarray:
    """União das categorias em resolução total (uint8, 0/255), com a borda suavizada pela ampliação."""
    H, W = om.size
    if not keys:
        return np.zeros((H, W), np.uint8)
    idx = [_ORDER.index(k) + 1 for k in keys]
    hit = np.isin(om.labels, idx).astype(np.float32)
    up = cv2.resize(hit, (W, H), interpolation=cv2.INTER_LINEAR)
    return ((up > 0.5).astype(np.uint8)) * 255


def _pieces(mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Componentes conexos (maiores que MIN_PIECE da foto): [(área, máscara bool)], maior primeiro."""
    H, W = mask.shape[:2]
    clean = cv2.morphologyEx((mask > 127).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    out = [(int(stats[i, cv2.CC_STAT_AREA]), lab == i) for i in range(1, n)
           if stats[i, cv2.CC_STAT_AREA] >= MIN_PIECE * H * W]
    out.sort(key=lambda t: -t[0])
    return out


def presentes(om: ObjectMap) -> list[tuple[str, float, int]]:
    """Categorias encontradas: [(chave, fração da foto, nº de objetos)], da maior para a menor."""
    out = []
    for i, k in enumerate(_ORDER):
        frac = float((om.labels == i + 1).mean())
        if frac < MIN_FRACTION:
            continue
        n = len(_pieces(category_mask(om, [k])))
        if n:
            out.append((k, frac, n))
    out.sort(key=lambda t: -t[1])
    return out


# ------------------------------------------------------------- refino (SAM)

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / u if u else 0.0


def refine_with_sam(mask: np.ndarray, emb, max_items: int = 12, min_iou: float = 0.55) -> np.ndarray:
    """Afia a borda de cada objeto da máscara usando o MobileSAM com um clique no centro dele.

    O resultado do SAM só vale se concordar com a máscara da rede (IoU ≥ min_iou) — assim ele não
    "engole" o vizinho nem pega só um pedaço — e nunca apaga o miolo da máscara da rede: ele só ajusta
    a faixa da borda. Caso contrário, mantém a máscara da rede para aquele objeto.
    Objetos pequenos demais são descartados (ruído)."""
    from . import segment
    H, W = mask.shape[:2]
    out = np.zeros((H, W), bool)
    r = max(3, int(round(0.015 * max(H, W))))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    rc = max(2, int(round(0.012 * max(H, W))))
    core_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rc + 1, 2 * rc + 1))
    for _, comp in _pieces(mask)[:max_items]:
        dt = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
        y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
        sam = segment.predict(emb, [((x + 0.5) / W, (y + 0.5) / H, 1)]) > 0
        if _iou(sam, comp) >= min_iou:
            near = cv2.dilate(comp.astype(np.uint8), kern) > 0
            core = cv2.erode(comp.astype(np.uint8), core_k) > 0
            out |= (sam & near) | core
        else:
            out |= comp
    return out.astype(np.uint8) * 255


def _smooth(mask: np.ndarray) -> np.ndarray:
    """Alisa a serrilhado da máscara (alpha 0–255) sem engordar."""
    s = max(0.7, 0.0012 * max(mask.shape[:2]))
    return cv2.GaussianBlur(mask, (0, 0), s)


def select(
    img: np.ndarray,
    manter: str | list[str] = "",
    apagar: str | list[str] = "",
    modelo: str = DEFAULT_MODEL,
    refinar: bool = True,
    base: np.ndarray | None = None,
    om: ObjectMap | None = None,
    emb=None,
) -> np.ndarray:
    """Máscara alpha (uint8 HxW, 255 = fica) pelos tipos de objeto escolhidos.

    manter: tipos que ficam. Se vazio, parte de `base` (a máscara da detecção principal).
    apagar: tipos que viram fundo (subtraídos do resultado).
    refinar: afia as bordas com o MobileSAM, se os modelos dele estiverem baixados."""
    from . import segment
    keep_keys = parse_keys(manter) if isinstance(manter, str) else list(manter)
    drop_keys = parse_keys(apagar) if isinstance(apagar, str) else list(apagar)
    both = set(keep_keys) & set(drop_keys)
    if both:
        raise ValueError(f"'{', '.join(sorted(both))}' está em manter e em apagar ao mesmo tempo.")
    if not keep_keys and base is None:
        raise ValueError("Sem objetos para manter, é preciso a detecção principal do Fundo como base.")
    om = om or scan(img, modelo)
    H, W = om.size
    use_sam = refinar and segment.model_available()
    if use_sam and emb is None and any(k in INSTANCE_KEYS for k in keep_keys + drop_keys):
        emb = segment.embed(img)

    def build(keys: list[str]) -> np.ndarray:
        inst = [k for k in keys if k in INSTANCE_KEYS]
        m = category_mask(om, [k for k in keys if k not in INSTANCE_KEYS])
        mi = category_mask(om, inst)
        if use_sam and mi.any():
            mi = refine_with_sam(mi, emb)
        return _smooth(np.maximum(m, mi))

    if keep_keys:
        out = build(keep_keys)
    else:
        out = cv2.resize(base, (W, H), interpolation=cv2.INTER_LINEAR) if base.shape[:2] != (H, W) else base.copy()
    if drop_keys:
        gone = build(drop_keys).astype(np.float32) / 255.0
        out = (out.astype(np.float32) * (1.0 - gone) + 0.5).astype(np.uint8)
    return np.ascontiguousarray(out)
