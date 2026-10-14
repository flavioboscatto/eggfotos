"""Gera os ícones do EggFotos a partir do logo (eggfotos_logo.jpg).

    python tools/gerar_icones.py [logo.jpg] [--padrao ambar|claro]

Saída em assets/icon/: EggFotos.icns (macOS), EggFotos.ico (Windows), EggFotos.png (1024 px, usado
pela janela e pelo Dock) e variantes em PNG. Duas versões: "claro" (fundo branco, fiel ao logo) e
"ambar" (cor de destaque do programa). Só o símbolo (ovo + câmera) entra no ícone: o texto
ficaria ilegível em tamanhos pequenos.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "assets" / "icon"
TAM = 1024
CORPO = 824                      # lado do quadrado arredondado (padrão da Apple em 1024)
INK = (29, 29, 33)               # tinta escura do desenho
VARIANTES = {
    "claro": {"fundo": (255, 255, 255), "tinta": (20, 20, 22)},
    "ambar": {"fundo": (242, 177, 52), "tinta": INK},
}


def extrair_simbolo(logo: Path) -> np.ndarray:
    """Devolve a opacidade (0–1) do desenho do símbolo, recortado, sem o texto abaixo."""
    gray = np.asarray(Image.open(logo).convert("L"), dtype=np.float32)
    ink = 1.0 - gray / 255.0
    h, w = ink.shape
    # o texto "EggFotos" fica na parte de baixo: corta na maior faixa vazia entre símbolo e texto
    rows = (ink > 0.5).sum(axis=1) > 0
    runs, start = [], None
    for y, on in enumerate(rows):
        if on and start is None:
            start = y
        if not on and start is not None:
            runs.append((start, y)); start = None
    if start is not None:
        runs.append((start, h))
    simbolo = max(runs, key=lambda r: r[1] - r[0])           # o bloco mais alto é o ovo com a câmera
    sub = ink[simbolo[0]:simbolo[1]]
    cols = np.flatnonzero((sub > 0.5).sum(axis=0))
    return sub[:, cols[0]: cols[-1] + 1]


def squircle(tam: int, lado: int, n: float = 5.0, ss: int = 4) -> np.ndarray:
    """Máscara (0–1) de uma superelipse centrada, com suavização por supersampling."""
    big = tam * ss
    y, x = np.mgrid[0:big, 0:big].astype(np.float32)
    c, r = (big - 1) / 2.0, lado * ss / 2.0
    inside = (np.abs((x - c) / r) ** n + np.abs((y - c) / r) ** n) <= 1.0
    m = Image.fromarray((inside * 255).astype(np.uint8)).resize((tam, tam), Image.LANCZOS)
    return np.asarray(m, dtype=np.float32) / 255.0


def montar(alpha_simbolo: np.ndarray, fundo, tinta) -> Image.Image:
    mask = squircle(TAM, CORPO)
    # sombra suave sob o ícone, como nos ícones do macOS
    sombra = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(14))
    sombra_a = np.asarray(sombra, dtype=np.float32) / 255.0 * 0.32
    sombra_a = np.roll(sombra_a, 10, axis=0)
    canvas = np.zeros((TAM, TAM, 4), np.float32)
    canvas[..., 3] = sombra_a                                 # sombra preta
    corpo = np.zeros((TAM, TAM, 3), np.float32)
    corpo[:] = fundo
    # símbolo centralizado (ocupa ~60% do corpo)
    h, w = alpha_simbolo.shape
    esc = (CORPO * 0.60) / max(h, w)
    sim = Image.fromarray((alpha_simbolo * 255).astype(np.uint8)).resize(
        (max(1, round(w * esc)), max(1, round(h * esc))), Image.LANCZOS)
    a = np.zeros((TAM, TAM), np.float32)
    sa = np.asarray(sim, dtype=np.float32) / 255.0
    y0, x0 = (TAM - sa.shape[0]) // 2, (TAM - sa.shape[1]) // 2
    a[y0:y0 + sa.shape[0], x0:x0 + sa.shape[1]] = sa
    corpo = corpo * (1 - a[..., None]) + np.array(tinta, np.float32) * a[..., None]
    m = mask[..., None]
    rgb = corpo * m + canvas[..., :3] * (1 - m)
    alpha = np.clip(mask + sombra_a * (1 - mask), 0, 1)
    out = np.dstack([rgb, alpha * 255]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def gravar(img: Image.Image, nome: str) -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    img.save(SAIDA / f"{nome}.png")
    for lado in (512, 256, 128, 64, 32, 16):
        img.resize((lado, lado), Image.LANCZOS).save(SAIDA / f"{nome}_{lado}.png")
    img.save(SAIDA / f"{nome}.ico", sizes=[(s, s) for s in (256, 128, 64, 48, 32, 16)])
    if sys.platform == "darwin":                              # .icns com o iconutil do macOS
        with tempfile.TemporaryDirectory() as tmp:
            iconset = Path(tmp) / f"{nome}.iconset"
            iconset.mkdir()
            for base in (16, 32, 128, 256, 512):
                img.resize((base, base), Image.LANCZOS).save(iconset / f"icon_{base}x{base}.png")
                img.resize((base * 2, base * 2), Image.LANCZOS).save(iconset / f"icon_{base}x{base}@2x.png")
            subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(SAIDA / f"{nome}.icns")], check=True)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    padrao = "ambar"
    if "--padrao" in sys.argv:
        padrao = sys.argv[sys.argv.index("--padrao") + 1]
        args = [a for a in args if a != padrao]
    if padrao not in VARIANTES:
        raise SystemExit(f"--padrao deve ser um de: {', '.join(VARIANTES)}")
    logo = Path(args[0]) if args else RAIZ / "assets" / "logo" / "eggfotos_logo.jpg"
    simbolo = extrair_simbolo(logo)
    print(f"símbolo recortado: {simbolo.shape[1]}×{simbolo.shape[0]} px")
    for nome, cores in VARIANTES.items():
        gravar(montar(simbolo, cores["fundo"], cores["tinta"]), f"EggFotos_{nome}")
    # a variante escolhida vira o ícone do programa (EggFotos.png/.ico/.icns)
    for ext in ("png", "ico", "icns"):
        origem = SAIDA / f"EggFotos_{padrao}.{ext}"
        if origem.exists():
            (SAIDA / f"EggFotos.{ext}").write_bytes(origem.read_bytes())
    print(f"ícone padrão: {padrao}")
    # símbolo sozinho, com transparência (para a tela "Sobre", site etc.)
    h, w = simbolo.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = INK
    rgba[..., 3] = (simbolo * 255).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(SAIDA / "EggFotos_simbolo.png")
    print("gerado em", SAIDA)


if __name__ == "__main__":
    main()
