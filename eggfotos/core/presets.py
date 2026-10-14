"""Perfis prontos para a remoção de marca d'água (combinações de controles que funcionam bem
em situações comuns) e perfis do próprio usuário, salvos em ~/.eggfotos/marcas/perfis.json.

Os valores seguem os parâmetros de inpaint.remove_with_mask (+ `sensibilidade`, usada no
"Apertar marcação aos traços"). São ponto de partida: refine com suas fotos e salve o seu perfil.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from . import watermark
from .inpaint import METHODS

CUSTOM = "Personalizado"
FIELDS = ("metodo", "expandir", "raio", "contexto", "emenda", "grao_forca", "sensibilidade")
RANGES = {"expandir": (0, 100), "raio": (1, 100), "contexto": (0, 100), "emenda": (0, 100),
          "grao_forca": (0, 200), "sensibilidade": (0, 100)}


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    dica: str
    metodo: str
    expandir: int
    raio: int
    contexto: int
    emenda: int
    grao_forca: int
    sensibilidade: int = 40

    def values(self) -> dict:
        return {f: getattr(self, f) for f in FIELDS}


PRESETS: dict[str, Preset] = {p.key: p for p in (
    Preset("liso", "Fundo liso (céu, parede, gradiente)",
           "Rápido e sem modelo. Funciona quando a área atrás da marca é lisa ou um degradê suave.",
           "telea", 30, 45, 40, 25, 100),
    Preset("texto_fino", "Marca fina de texto (letras)",
           "Marque a área de forma solta e use \"Apertar marcação aos traços\" antes de remover: "
           "menos área para a IA inventar.",
           "lama", 40, 50, 40, 25, 100, 40),
    Preset("textura", "Textura / natureza (folhas, tecido, pele)",
           "LaMa com mais contexto e um pouco mais de grão, para a emenda não parecer lisa.",
           "lama", 25, 50, 60, 30, 110),
    Preset("detalhes", "Perto de rostos, bordas ou objetos",
           "Máximo contexto para a IA respeitar o que está em volta. Mais lento: confira a 100%.",
           "lama", 20, 50, 75, 35, 100),
    Preset("grande", "Marca grande (mais de ~900 px)",
           "Faça em 2 ou 3 passadas: marque um pedaço, remova, confira e repita. Poucas janelas por vez.",
           "lama", 15, 50, 30, 35, 100),
    Preset("retangulo", "Retângulo solto (sem apertar)",
           "Para quando você marcou uma caixa larga em volta da marca: não engorda a marcação.",
           "lama", 0, 50, 50, 30, 100),
    Preset("grao_alto", "Foto com muito grão (ISO alto)",
           "Recoloca mais grão no trecho preenchido, para combinar com a foto.",
           "lama", 30, 50, 40, 25, 140),
    Preset("foto_limpa", "Foto limpa ou muito comprimida",
           "Pouco grão no trecho preenchido, para não parecer sujeira numa foto lisa.",
           "lama", 30, 50, 40, 25, 60),
)}


def preset_keys() -> list[str]:
    return list(PRESETS)


# ------------------------------------------------------------ perfis do usuário

def _file():
    return watermark.marks_dir() / "perfis.json"


def _clean(values: dict) -> dict:
    out = {"metodo": values.get("metodo", "telea")}
    if out["metodo"] not in METHODS:
        raise ValueError(f"Método inválido: {out['metodo']}")
    for f, (lo, hi) in RANGES.items():
        out[f] = int(max(lo, min(hi, round(float(values.get(f, PRESET_DEFAULTS[f]))))))
    return out


PRESET_DEFAULTS = {"expandir": 30, "raio": 50, "contexto": 40, "emenda": 20, "grao_forca": 100,
                   "sensibilidade": 40}


def load_user_profiles() -> dict[str, dict]:
    try:
        raw = json.loads(_file().read_text(encoding="utf-8"))
        return {str(n): _clean(v) for n, v in raw.items()}
    except (OSError, ValueError, AttributeError):
        return {}


def save_user_profile(nome: str, values: dict) -> str:
    nome = nome.strip()
    if not nome or nome == CUSTOM:
        raise ValueError("Dê um nome ao perfil (diferente de \"Personalizado\").")
    if any(p.label == nome for p in PRESETS.values()):
        raise ValueError("Já existe um perfil pronto com esse nome. Escolha outro.")
    perfis = load_user_profiles()
    perfis[nome] = _clean(values)
    _file().write_text(json.dumps(perfis, ensure_ascii=False, indent=1), encoding="utf-8")
    return nome


def delete_user_profile(nome: str) -> None:
    perfis = load_user_profiles()
    if perfis.pop(nome, None) is not None:
        _file().write_text(json.dumps(perfis, ensure_ascii=False, indent=1), encoding="utf-8")
