"""Pré-análise da foto para a remoção de marca d'água, SEM IA (só medidas da imagem).

Duas coisas independentes:
  1. suggest_settings(img, mask) -> Suggestion   Mede a vizinhança da área já marcada (textura, grão,
     espessura dos traços, tamanho, rostos por perto) e sugere os controles da aba Marca d'água.
  2. find_candidates(img) -> Candidates          Procura REGIÕES que parecem marca d'água (traços finos e
     coerentes, de preferência perto das bordas). É um palpite: o usuário confere e corrige.

Tudo é relativo ao tamanho da foto (a mesma foto em outra resolução dá a mesma sugestão) e
determinístico. Funções puras: não alteram a imagem de entrada.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import faces
from .imgutils import split_alpha
from .inpaint import _bbox, _noise_std, snap_to_strokes
from .presets import PRESETS, RANGES

ANALYSIS_SIDE = 400   # lado maior (px) do recorte onde se mede a textura: independe da resolução


@dataclass
class Suggestion:
    values: dict                       # controles: metodo, expandir, raio, contexto, emenda, grao_forca, sensibilidade
    polaridade: str                    # "clara" | "escura" | "auto"
    apertar: bool                      # vale usar "Apertar marcação aos traços"
    perfil: str                        # chave do perfil pronto mais parecido
    motivos: list[str] = field(default_factory=list)
    medidas: dict = field(default_factory=dict)

    def resumo(self) -> str:
        return "\n".join("• " + m for m in self.motivos)


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)


def _texture(gray: np.ndarray, ring: np.ndarray) -> tuple[float, float]:
    """(gradiente médio, fração de pixels de borda forte) no anel, numa cópia reduzida."""
    s = min(1.0, ANALYSIS_SIDE / max(gray.shape[:2]))
    g, r = gray, ring
    if s < 1.0:
        size = (max(1, round(gray.shape[1] * s)), max(1, round(gray.shape[0] * s)))
        g = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        r = cv2.resize(ring.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.2)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mag = np.hypot(gx, gy)[r]
    if mag.size < 30:
        return 0.0, 0.0
    return float(mag.mean()), float((mag > 10.0).mean())


def _faces_near(img: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """Há rosto encostado na área? Só tenta se o modelo de rostos já estiver baixado."""
    try:
        if not faces.model_available():
            return False
        H, W = img.shape[:2]
        x0, y0, x1, y1 = box
        gw, gh = (x1 - x0) * 0.6, (y1 - y0) * 0.6
        for fx, fy, fw, fh, score in faces.detect_faces(img):
            if score < 0.7:       # textura confunde o detector; só rosto com boa confiança conta
                continue
            fx, fy, fw, fh = fx * W, fy * H, fw * W, fh * H
            if fx < x1 + gw and fx + fw > x0 - gw and fy < y1 + gh and fy + fh > y0 - gh:
                return True
    except Exception:  # modelo quebrado/ausente: a sugestão segue sem essa informação
        return False
    return False


def _clamp(values: dict) -> dict:
    out = dict(values)
    for k, (lo, hi) in RANGES.items():
        if k in out:
            out[k] = int(max(lo, min(hi, round(out[k]))))
    return out


def suggest_settings(img: np.ndarray, mask: np.ndarray, lama_disponivel: bool = True) -> Suggestion:
    """Sugere os controles de preenchimento para a área marcada (`mask` uint8, >127 = marcado)."""
    if mask.shape[:2] != img.shape[:2]:
        raise ValueError("A máscara deve ter o mesmo tamanho da imagem.")
    box = _bbox(mask > 127)
    if box is None:
        raise ValueError("Marque a área da marca d'água antes de pedir a sugestão.")
    rgb, _ = split_alpha(img)
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    ref = float(max(H, W))

    # recorte em volta da marca (o resto da foto não interessa)
    pad = int(max(24, 0.6 * max(bw, bh)))
    cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    gray = _gray(rgb[cy0:cy1, cx0:cx1])
    m = mask[cy0:cy1, cx0:cx1] > 127
    r_ring = max(6, int(0.3 * max(bw, bh)))
    dist = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)
    ring = (dist > 0) & (dist <= r_ring)
    if ring.sum() < 50:                      # marcação colada nas bordas ou cobrindo quase tudo
        ring = ~m
    mag_medio, bordas = _texture(gray, ring)
    grao = _noise_std(gray, ring)

    # geometria da marcação
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    largura = 2.0 * float(np.percentile(dt[m], 90))             # espessura típica dos traços (px)
    rel_traco = largura / ref
    preenchimento = float(m.sum()) / max(1.0, float(bw * bh))
    tamanho_rel = max(bw, bh) / ref
    area_rel = float(m.sum()) / (H * W)
    apertar = preenchimento >= 0.55 and rel_traco > 0.006

    perto_rosto = _faces_near(rgb, box)

    nivel = ("liso" if mag_medio < 1.2 and bordas < 0.02
             else "alta" if mag_medio > 5.0 or bordas > 0.15 else "media")
    grande = tamanho_rel > 0.5 or area_rel > 0.08
    fino = rel_traco < 0.006

    # --------------------------------------------------------------- decisões
    motivos: list[str] = []
    if nivel == "liso" and not perto_rosto and not grande:
        metodo = "telea"
        motivos.append("O fundo ao redor é liso (céu, parede ou degradê): o preenchimento rápido basta e não precisa de modelo.")
    else:
        metodo = "lama"
        if perto_rosto:
            motivos.append("Há um rosto perto da marca: IA (LaMa) com bastante contexto para respeitar o rosto.")
        elif grande:
            motivos.append("A marca é grande: IA (LaMa) com contexto menor, para não ficar lenta. "
                           "Se ficar estranho, faça em 2 ou 3 passadas.")
        elif nivel == "alta":
            motivos.append("O fundo tem muita textura ou bordas: a IA (LaMa) reconstrói isso melhor que o preenchimento rápido.")
        else:
            motivos.append("O fundo tem textura moderada: IA (LaMa) dá emenda mais natural.")
        if not lama_disponivel:
            motivos.append("O modelo LaMa ainda não está baixado — baixe-o no botão da seção 2 para usar esta sugestão.")

    expandir = 40 if fino else 35 if apertar else 15 if preenchimento >= 0.55 else 30
    raio = 35 if fino else 60 if rel_traco > 0.02 else 50
    contexto = 75 if perto_rosto else 30 if grande else {"liso": 30, "media": 50, "alta": 60}[nivel]
    emenda = 35 if (perto_rosto or grande) else 30 if nivel == "alta" else 20
    if grao < 0.9:
        grao_forca = 60
        motivos.append("A foto é bem limpa (quase sem grão): coloco pouco grão no trecho preenchido.")
    elif grao > (5.0 if nivel == "alta" else 3.0):   # em fundo texturado a textura infla a medida
        grao_forca = 140
        motivos.append("A foto tem bastante grão: recoloco mais grão no trecho preenchido para combinar.")
    else:
        grao_forca = 100

    if fino:
        motivos.append("Os traços da marca são finos (letras): expando a marcação um pouco mais para cobrir o contorno suave.")
    if apertar:
        motivos.append("A marcação está solta (área cheia): use \"Apertar marcação aos traços\" antes de remover — "
                       "a IA terá menos área para inventar.")

    # polaridade: a marca é mais clara ou mais escura que o que está em volta dos traços?
    polaridade = "auto"
    hit = snap_to_strokes(rgb, mask, "auto", 40)
    if hit.any():
        hm = hit[cy0:cy1, cx0:cx1] > 0
        around = (cv2.dilate(hm.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ~hm & m
        if around.sum() > 30 and hm.sum() > 10:
            diff = float(gray[hm].mean()) - float(gray[around].mean())
            if abs(diff) > 4:
                polaridade = "clara" if diff > 0 else "escura"
                motivos.append(f"A marca parece {'mais clara' if diff > 0 else 'mais escura'} que o fundo: "
                               f"deixo \"{'Marca clara' if diff > 0 else 'Marca escura'}\" no ajuste aos traços.")

    # perfil pronto mais parecido (só para o usuário se situar)
    if metodo == "telea":
        perfil = "liso"
    elif perto_rosto:
        perfil = "detalhes"
    elif grande:
        perfil = "grande"
    elif nivel == "alta":
        perfil = "textura"
    elif grao_forca == 140:
        perfil = "grao_alto"
    elif grao_forca == 60:
        perfil = "foto_limpa"
    elif fino:
        perfil = "texto_fino"
    else:
        perfil = "retangulo" if preenchimento >= 0.55 and not apertar else "textura"
    motivos.append(f"Perfil pronto mais parecido: {PRESETS[perfil].label}.")

    values = _clamp({"metodo": metodo, "expandir": expandir, "raio": raio, "contexto": contexto,
                     "emenda": emenda, "grao_forca": grao_forca, "sensibilidade": 40})
    values["metodo"] = metodo
    medidas = {"textura": round(mag_medio, 2), "bordas": round(bordas, 3), "grao": round(grao, 2),
               "largura_traco_rel": round(rel_traco, 4), "preenchimento": round(preenchimento, 2),
               "tamanho_rel": round(tamanho_rel, 3), "rosto_perto": perto_rosto, "nivel": nivel}
    return Suggestion(values, polaridade, apertar, perfil, motivos, medidas)


# ============================================================ procurar candidatas

TIPOS = {"texto": "Texto", "logotipo": "Logotipo"}


@dataclass
class Candidate:
    x: float
    y: float
    w: float
    h: float            # retângulo RELATIVO (0–1)
    score: float
    polaridade: str     # "clara" | "escura"
    lugar: str          # descrição curta ("canto inferior direito"…)
    tipo: str = "texto"  # "texto" | "logotipo"

    def rect(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.w, self.h

    def rotulo(self) -> str:
        return f"{TIPOS.get(self.tipo, self.tipo)} · {self.lugar}"


@dataclass
class Candidates:
    itens: list[Candidate]
    repetida: bool = False     # padrão repetido pela foto toda (marca em mosaico)


def _place(x: float, y: float, w: float, h: float) -> str:
    cx, cy = x + w / 2, y + h / 2
    col = "esquerdo" if cx < 0.33 else "direito" if cx > 0.67 else "central"
    lin = "superior" if cy < 0.33 else "inferior" if cy > 0.67 else "central"
    if lin == "central" and col == "central":
        return "centro da foto"
    if lin == "central":
        return f"lado {col}"
    if col == "central":
        return f"parte {lin}"
    return f"canto {lin} {col}"


def _hat_maps(gray: np.ndarray, kernel_rel: float, t: float, kmin: int = 5, kmax: int = 15,
              blur: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Estruturas claras e escuras MENORES que o núcleo (top-hat / black-hat) acima do limiar `t`."""
    side = max(gray.shape[:2])
    k = int(np.clip(round(kernel_rel * side), kmin, kmax)) | 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    g = cv2.GaussianBlur(gray, (0, 0), blur)
    return (cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kern) > t), (cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kern) > t)


def _stroke_maps(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Traços finos claros e escuros (letras), com limiar fixo em níveis de cinza."""
    return _hat_maps(gray, 0.014, 10)


def _repeated(s: np.ndarray) -> bool:
    """Autocorrelação da mapa de traços: pico fora do centro = padrão repetido (marca em mosaico)."""
    f = s.astype(np.float32)
    cover = float(f.mean())
    if not 0.01 < cover < 0.4:
        return False
    h0, w0 = s.shape
    cells = [float(f[i * h0 // 3:(i + 1) * h0 // 3, j * w0 // 3:(j + 1) * w0 // 3].mean())
             for i in range(3) for j in range(3)]
    if sum(c > 0.02 for c in cells) < 6:   # um texto só num canto não é "marca repetida"
        return False
    f -= cover
    spec = np.fft.rfft2(f)
    ac = np.fft.irfft2(spec * np.conj(spec), s=f.shape)
    ac /= max(float(ac[0, 0]), 1e-6)
    h, w = ac.shape
    yy = np.minimum(np.arange(h), h - np.arange(h))[:, None]
    xx = np.minimum(np.arange(w), w - np.arange(w))[None, :]
    far = np.hypot(yy, xx) > 0.08 * max(h, w)
    return float(ac[far].max()) > 0.3


def _border_bonus(x0: float, y0: float, x1: float, y1: float, w: int, h: int) -> float:
    """1,0 no meio da foto, até 1,6 colado numa borda (marcas ficam perto das bordas)."""
    d = min(x0 / w, 1 - x1 / w, y0 / h, 1 - y1 / h)
    return 1.0 + 0.6 * float(np.clip(1.0 - d / 0.25, 0, 1))


def _text_candidates(gray: np.ndarray, max_n: int) -> list[Candidate]:
    """Texto e traços finos: concentração local de traços (top-hat/black-hat fino)."""
    h, w = gray.shape
    side = max(h, w)
    win = max(15, int(0.16 * side)) | 1
    claras, escuras = _stroke_maps(gray)
    achados: list[Candidate] = []
    for pol, S in (("clara", claras), ("escura", escuras)):
        S8 = S.astype(np.uint8)
        dens = cv2.boxFilter(S8.astype(np.float32), -1, (win, win), normalize=True,
                             borderType=cv2.BORDER_CONSTANT)
        base = float(np.median(dens))
        score = dens - base
        yy, xx = np.mgrid[0:h, 0:w]
        borda = np.minimum.reduce([xx / w, 1 - xx / w, yy / h, 1 - yy / h])
        score = score * (1.0 + 0.6 * np.clip(1.0 - borda / 0.25, 0, 1))
        # junta os traços vizinhos (letras da mesma palavra/linha) para achar a caixa inteira
        kj = int(np.clip(round(0.03 * side), 5, 41)) | 1
        grupos = cv2.morphologyEx(S8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kj, kj)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(grupos, connectivity=8)
        nr, lab_raw, st_raw, _ = cv2.connectedComponentsWithStats(S8, connectivity=8)
        taken = np.zeros((h, w), bool)
        for _ in range(max_n):
            masked = np.where(taken, -1.0, score)
            iy, ix = np.unravel_index(int(np.argmax(masked)), masked.shape)
            v = float(masked[iy, ix])
            # tem de se destacar do resto da foto: em foto muito texturada tudo "parece" marca
            if v < 0.04 or dens[iy, ix] < 0.03 or dens[iy, ix] < 2.5 * base + 0.02:
                break
            r = win // 2
            ya, yb, xa, xb = max(0, iy - r), min(h, iy + r + 1), max(0, ix - r), min(w, ix + r + 1)
            ids = {int(i) for i in np.unique(lab[ya:yb, xa:xb][S8[ya:yb, xa:xb] > 0]) if i > 0}
            taken[max(0, iy - win):iy + win, max(0, ix - win):ix + win] = True
            ids = [i for i in ids if st[i, cv2.CC_STAT_AREA] >= max(8, 0.00002 * h * w)
                   and st[i, cv2.CC_STAT_WIDTH] < 0.7 * w and st[i, cv2.CC_STAT_HEIGHT] < 0.5 * h]
            if not ids:
                continue
            bx0 = min(st[i, cv2.CC_STAT_LEFT] for i in ids)
            by0 = min(st[i, cv2.CC_STAT_TOP] for i in ids)
            bx1 = max(st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH] for i in ids)
            by1 = max(st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] for i in ids)
            m = int(0.008 * side) + 2    # uma folga em volta dos traços
            rx0, ry0 = max(0, bx0 - m), max(0, by0 - m)
            rx1, ry1 = min(w, bx1 + m), min(h, by1 + m)
            if rx1 - rx0 < 6 or ry1 - ry0 < 4:
                continue
            # várias peças pequenas e alinhadas = letras; uma peça só (ou poucas) = símbolo
            letras = [j for j in range(1, nr) if st_raw[j, cv2.CC_STAT_AREA] >= 4
                      and bx0 <= st_raw[j, cv2.CC_STAT_LEFT] <= bx1 and by0 <= st_raw[j, cv2.CC_STAT_TOP] <= by1]
            largo = (bx1 - bx0) >= 2.2 * (by1 - by0)       # texto corre em linha; selo/símbolo é mais quadrado
            tipo = "texto" if len(letras) >= 4 and largo else "logotipo"
            x, y, cw, ch = rx0 / w, ry0 / h, (rx1 - rx0) / w, (ry1 - ry0) / h
            achados.append(Candidate(x, y, cw, ch, v, pol, _place(x, y, cw, ch), tipo))
    return achados


def _logo_candidates(gray: np.ndarray) -> list[Candidate]:
    """Logotipos e selos: formas maiores e sólidas (claras ou escuras), isoladas, perto das bordas.

    Top-hat/black-hat com núcleos grandes pegam o que é mais claro/escuro que a vizinhança e menor
    que o núcleo. Só vale se a forma for compacta, preenchida e a área em volta for calma."""
    h, w = gray.shape
    side = max(h, w)
    out: list[Candidate] = []
    cobertura = []
    mapas = []
    for pol, idx in (("clara", 0), ("escura", 1)):
        S = np.zeros((h, w), bool)
        for kr in (0.05, 0.12):
            S |= _hat_maps(gray, kr, 14, kmin=7, kmax=101, blur=1.0)[idx]
        S8 = cv2.morphologyEx(S.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cobertura.append(float(S8.mean()))
        mapas.append((pol, S8))
    if max(cobertura) > 0.12:         # foto cheia de formas: nada se destaca
        return out
    kj = int(np.clip(round(0.025 * side), 5, 31)) | 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kj, kj))
    for pol, S8 in mapas:
        grupos = cv2.morphologyEx(S8, cv2.MORPH_CLOSE, kern)
        n, lab, st, _ = cv2.connectedComponentsWithStats(grupos, connectivity=8)
        for i in range(1, n):
            x0, y0 = int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP])
            bw, bh = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
            frac = bw * bh / float(h * w)
            if not 0.0015 < frac < 0.18 or bw > 0.5 * w or bh > 0.5 * h:
                continue
            box = S8[y0:y0 + bh, x0:x0 + bw]
            fill = float(box.mean())
            if fill < 0.12:
                continue
            m = int(max(bw, bh) * 0.5) + 2
            ya, yb, xa, xb = max(0, y0 - m), min(h, y0 + bh + m), max(0, x0 - m), min(w, x0 + bw + m)
            anel = S8[ya:yb, xa:xb].astype(np.float32)
            area_anel = (yb - ya) * (xb - xa) - bw * bh
            dens_anel = (float(anel.sum()) - float(box.sum())) / max(1, area_anel)
            isolamento = fill - dens_anel
            if isolamento < 0.15:
                continue
            mg = int(0.006 * side) + 2
            rx0, ry0, rx1, ry1 = max(0, x0 - mg), max(0, y0 - mg), min(w, x0 + bw + mg), min(h, y0 + bh + mg)
            score = isolamento * _border_bonus(rx0, ry0, rx1, ry1, w, h) * 0.5
            x, y, cw, ch = rx0 / w, ry0 / h, (rx1 - rx0) / w, (ry1 - ry0) / h
            out.append(Candidate(x, y, cw, ch, float(score), pol, _place(x, y, cw, ch), "logotipo"))
    return out


def _contained(a: Candidate, b: Candidate, frac: float = 0.7) -> bool:
    """`a` está (quase) todo dentro de `b`?"""
    ix = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    return ix * iy >= frac * a.w * a.h


def find_candidates(img: np.ndarray, max_n: int = 5) -> Candidates:
    """Procura textos e logotipos que parecem marca d'água: traços finos coerentes (texto) e formas
    sólidas isoladas (logotipo/selo), claros OU escuros, de preferência perto das bordas.
    Sem IA: serve como ponto de partida, nunca como verdade — o usuário aceita ou descarta."""
    rgb, _ = split_alpha(img)
    H, W = rgb.shape[:2]
    s = min(1.0, 800.0 / max(H, W))
    small = cv2.resize(np.ascontiguousarray(rgb), (max(1, round(W * s)), max(1, round(H * s))),
                       interpolation=cv2.INTER_AREA) if s < 1.0 else np.ascontiguousarray(rgb)
    gray = _gray(small)
    claras, escuras = _stroke_maps(gray)
    repetida = _repeated(claras | escuras)

    # os logotipos são medidos numa cópia ainda menor (núcleos grandes ficam rápidos)
    g2 = gray if max(gray.shape) <= 400 else cv2.resize(
        gray, (max(1, round(gray.shape[1] * 400 / max(gray.shape))), max(1, round(gray.shape[0] * 400 / max(gray.shape)))),
        interpolation=cv2.INTER_AREA)
    textos = sorted(_text_candidates(gray, max_n), key=lambda c: -c.score)
    logos = sorted(_logo_candidates(g2), key=lambda c: -c.score)
    out: list[Candidate] = []
    for c in textos:
        if all(_iou(c, o) < 0.4 and not _contained(c, o) for o in out):
            out.append(c)
    for c in logos:                                 # texto achado tem prioridade sobre o mesmo trecho visto como logotipo
        iguais = [o for o in out if _iou(c, o) >= 0.4 or _contained(c, o)]
        if iguais:
            continue
        miudos = [o for o in out if _contained(o, c) and o.w * o.h * 2.5 <= c.w * c.h]
        out = [o for o in out if o not in miudos]   # um logotipo que engloba textos menores os substitui
        if any(_contained(o, c) for o in out):
            continue
        out.append(c)
    out.sort(key=lambda c: -c.score)
    out = out[:max_n]
    return Candidates(out, repetida)


def _iou(a: Candidate, b: Candidate) -> float:
    ix = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0
