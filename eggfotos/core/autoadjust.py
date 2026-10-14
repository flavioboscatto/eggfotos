"""Ajuste automático ("varinha mágica") SEM IA: mede a foto e sugere os controles da aba Ajustes.

suggest_adjustments(img) -> AutoSuggestion   (valores no mesmo formato de adjustments.adjust)

Medidas (todas em cima de uma cópia reduzida, exceto o ruído, que precisa de pixels reais):
  exposição  mediana da luminância → alvo ~0,46 (em passos, com cautela em cena noturna)
  contraste  faixa entre os percentis 1 e 99 (foto "lavada" ganha contraste)
  sombras    fração de pixels bem escuros (recupera detalhe)
  realces    fração de pixels estourados (puxa de volta)
  saturação  croma mediano dos pixels coloridos
  temperatura  dominante de cor medida só nos pixels quase neutros (cinza-mundo seletivo)
  nitidez    variância do laplaciano (foto mole ganha nitidez)
  granulação ruído estimado (Immerkaer) em pixels reais

É uma SUGESTÃO conservadora: o resultado vai para os sliders e o usuário refina. Determinística.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .adjustments import ADJUSTMENT_KEYS, adjustment_range, neutral_params
from .imgutils import luminance, split_alpha, to_float

ANALYSIS_SIDE = 512
TARGET_MEDIAN = 0.46


@dataclass
class AutoSuggestion:
    values: dict[str, int]
    motivos: list[str] = field(default_factory=list)
    medidas: dict = field(default_factory=dict)

    def scaled(self, intensidade: float) -> dict[str, int]:
        """Os mesmos ajustes com a intensidade pedida (100 = como sugerido, 0 = nada)."""
        f = max(0.0, float(intensidade)) / 100.0
        out = {}
        for k, v in self.values.items():
            lo, hi = adjustment_range(k)
            out[k] = int(max(lo, min(hi, round(v * f))))
        return out

    def is_empty(self) -> bool:
        return not any(self.values.values())


def _small(rgb: np.ndarray) -> np.ndarray:
    H, W = rgb.shape[:2]
    s = ANALYSIS_SIDE / max(H, W)
    if s >= 1.0:
        return np.ascontiguousarray(rgb)
    return cv2.resize(np.ascontiguousarray(rgb), (max(1, round(W * s)), max(1, round(H * s))),
                      interpolation=cv2.INTER_AREA)


def estimate_noise(rgb: np.ndarray) -> float:
    """Desvio do ruído (níveis de cinza, 0–255) pelo método de Immerkaer, numa janela central em
    pixels reais. Robusto a bordas; fotos lisas e ruidosas dão o valor mais confiável."""
    H, W = rgb.shape[:2]
    c = 640
    y0, x0 = max(0, (H - c) // 2), max(0, (W - c) // 2)
    g = cv2.cvtColor(np.ascontiguousarray(rgb[y0:y0 + c, x0:x0 + c]), cv2.COLOR_RGB2GRAY).astype(np.float32)
    if min(g.shape) < 16:
        return 0.0
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = np.abs(cv2.filter2D(g, -1, k)[1:-1, 1:-1])
    # descarta o quarto mais "borda" para a textura não ser confundida com ruído
    r = r[r <= np.percentile(r, 75)]
    return float(np.sqrt(np.pi / 2.0) / 6.0 * r.mean() * 1.25) if r.size else 0.0


def suggest_adjustments(img: np.ndarray) -> AutoSuggestion:
    """Mede a foto e sugere os ajustes (ver o módulo)."""
    rgb, _ = split_alpha(img)
    sm = _small(rgb)
    x = to_float(sm)
    lum = luminance(x)
    p1, p5, p50, p95, p99 = (float(v) for v in np.percentile(lum, (1, 5, 50, 95, 99)))
    mot: list[str] = []
    v = neutral_params()

    # ---- exposição: leva a mediana para perto do alvo, com cautela em cena escura de propósito
    med = max(p50, 0.02)
    passos = float(np.log2(TARGET_MEDIAN / med))
    cautela = 0.6
    if med < 0.2 and p95 < 0.65:          # noite / cena escura: não "vira dia"
        cautela = 0.35
    if abs(passos * cautela) >= 0.12:
        v["exposicao"] = int(np.clip(round(passos * cautela * 50), -60, 60))
        mot.append(f"A foto está {'escura' if passos > 0 else 'clara demais'}: "
                   f"{'aumento' if passos > 0 else 'reduzo'} a exposição.")

    # ---- contraste: faixa tonal curta = foto "lavada"
    faixa = p99 - p1
    if faixa < 0.80:
        v["contraste"] = int(np.clip(round((0.92 / max(faixa, 0.25) - 1.0) * 100 * 0.5), 0, 40))
        if v["contraste"] >= 6:
            mot.append("As cores estão \"lavadas\" (pouca diferença entre claros e escuros): aumento o contraste.")
        else:
            v["contraste"] = 0
    elif faixa > 0.97 and p5 < 0.03 and p95 > 0.97:
        v["contraste"] = -8
        mot.append("Há muito contraste, com escuros e claros no limite: suavizo um pouco o contraste.")

    # ---- sombras e realces
    escuros = float((lum < 0.12).mean())
    if escuros > 0.15 and p50 > 0.18:
        v["sombras"] = int(np.clip(round(escuros * 120), 8, 50))
        mot.append("Há áreas muito escuras escondendo detalhe: levanto as sombras.")
    estourados = float((lum > 0.97).mean())
    if estourados > 0.015:
        v["realces"] = -int(np.clip(round(estourados * 600), 10, 40))   # branco já estourado não volta: só atenua
        mot.append("Há áreas estouradas (brancas demais): recupero os realces.")

    # ---- saturação: croma mediano dos pixels que têm cor
    hsv = cv2.cvtColor(sm, cv2.COLOR_RGB2HSV).astype(np.float32)
    s_ch, v_ch = hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    colorido = (v_ch > 0.15) & (s_ch > 0.05)
    sat = float(np.median(s_ch[colorido])) if colorido.mean() > 0.05 else None
    if sat is not None:
        delta = int(np.clip(round((0.34 - sat) / 0.34 * 60), -15, 35))
        if abs(delta) >= 8:
            v["saturacao"] = delta
            mot.append("As cores estão " + ("apagadas: aumento a saturação." if delta > 0
                                            else "muito fortes: reduzo um pouco a saturação."))

    # ---- temperatura: dominante de cor nos pixels QUASE neutros (um céu azul não é "dominante")
    neutro = (s_ch < 0.22) & (v_ch > 0.2) & (v_ch < 0.92)
    if neutro.mean() > 0.04:
        r_m, b_m = float(x[..., 0][neutro].mean()), float(x[..., 2][neutro].mean())
        cast = float(np.log(max(r_m, 1e-3) / max(b_m, 1e-3)))
        if abs(cast) > 0.07:
            # o slider multiplica R por (1+0,12t) e B por (1−0,12t): corrige 60% da dominante medida
            v["temperatura"] = int(np.clip(round(100 * np.tanh(-cast * 0.6 / 2) / 0.12), -40, 40))
            mot.append("Os tons neutros estão " + ("azulados: esquento a foto." if cast < 0
                                                   else "amarelados: esfrio a foto."))

    # ---- nitidez: laplaciano numa cópia suavizada (o ruído não conta como "detalhe")
    g = cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY), (0, 0), 1.0).astype(np.float32)
    lap = float(cv2.Laplacian(g, cv2.CV_32F).var())
    if lap < 70:
        v["nitidez"] = 40 if lap < 15 else 28 if lap < 35 else 15
        mot.append("A foto está um pouco mole: aumento a nitidez.")

    # ---- granulação
    ruido = estimate_noise(rgb)
    if ruido > 3.5:
        v["ruido"] = int(np.clip(round((ruido - 3.0) * 7), 8, 60))
        mot.append("Há granulação (ruído): ativo a remoção de granulação — confira em 100%.")
        if v["nitidez"] > 15:
            v["nitidez"] = 15      # afiar ruído só piora

    for k in ADJUSTMENT_KEYS:
        lo, hi = adjustment_range(k)
        v[k] = int(max(lo, min(hi, v[k])))
    if not any(v.values()):
        mot.append("A foto já está equilibrada: não vejo o que melhorar automaticamente.")
    medidas = {"mediana": round(p50, 3), "faixa": round(faixa, 3), "escuros": round(escuros, 3),
               "estourados": round(estourados, 3), "saturacao": None if sat is None else round(sat, 3),
               "laplaciano": round(lap, 1), "ruido": round(ruido, 2)}
    return AutoSuggestion(v, mot, medidas)
