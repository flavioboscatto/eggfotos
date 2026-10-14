"""Registro de operações reproduzíveis.

Cada operação é uma função pura `(imagem, **parâmetros) -> imagem` com um
esquema de parâmetros. A edição manual e o lote usam as MESMAS funções:
a interface monta o dicionário de parâmetros; o lote o lê da receita.

Um "step" é um dicionário: {"op": "<chave>", <parâmetros>...}

Para adicionar uma ferramenta nova ao lote (ex.: remover fundo):
escreva a função no core e registre-a aqui com register(Operation(...)).
O diálogo de lote monta o formulário sozinho a partir dos Params.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..core import adjustments, background, blur, faces, filters, frames, geometry, inpaint, objects, watermark


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    kind: str  # "int" | "float" | "choice" | "bool" | "color"
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[tuple[Any, str], ...] = ()  # (valor, rótulo)

    def coerce(self, value: Any) -> Any:
        if self.kind == "int":
            v = int(round(float(value)))
        elif self.kind == "float":
            v = float(value)
        elif self.kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "sim", "yes")
            return bool(value)
        elif self.kind == "choice":
            valid = [c[0] for c in self.choices]
            # aceita "90" e 90 indistintamente
            for c in valid:
                if str(c) == str(value):
                    return c
            raise ValueError(f"'{value}' não é opção válida para {self.label}")
        else:
            return str(value)
        if self.minimum is not None and v < self.minimum:
            raise ValueError(f"{self.label}: mínimo {self.minimum}")
        if self.maximum is not None and v > self.maximum:
            raise ValueError(f"{self.label}: máximo {self.maximum}")
        return v


@dataclass(frozen=True)
class Operation:
    key: str
    label: str
    func: Callable[..., np.ndarray]
    params: tuple[Param, ...] = field(default_factory=tuple)
    description: str = ""
    check: Callable[[dict], str | None] | None = None  # valida o passo (ex.: arquivo existe); devolve o erro

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def new_step(self, **overrides) -> dict:
        step = {"op": self.key, **self.defaults()}
        step.update(overrides)
        return step


REGISTRY: dict[str, Operation] = {}


def register(op: Operation) -> Operation:
    REGISTRY[op.key] = op
    return op


def get_operation(key: str) -> Operation:
    try:
        return REGISTRY[key]
    except KeyError:
        raise ValueError(f"Operação desconhecida: {key}") from None


def normalize_step(step: dict) -> dict:
    """Valida o step, preenche parâmetros ausentes e converte tipos."""
    if "op" not in step:
        raise ValueError("Passo sem o campo 'op'.")
    op = get_operation(step["op"])
    names = {p.name for p in op.params}
    unknown = set(step) - names - {"op"}
    if unknown:
        raise ValueError(f"{op.label}: parâmetro desconhecido {', '.join(sorted(unknown))}")
    out = {"op": op.key}
    for p in op.params:
        out[p.name] = p.coerce(step.get(p.name, p.default))
    return out


def apply_step(img: np.ndarray, step: dict) -> np.ndarray:
    step = normalize_step(step)
    op = REGISTRY[step["op"]]
    params = {k: v for k, v in step.items() if k != "op"}
    return op.func(img, **params)


def _objs_text(s: dict) -> str:
    """', manter: pessoas; apagar: veículos' (vazio se o passo não usa tipos de objeto)."""
    parts = []
    for campo, rot in (("manter", "manter"), ("apagar", "apagar")):
        try:
            keys = objects.parse_keys(s.get(campo, ""))
        except ValueError:
            keys = [str(s.get(campo, ""))]
        if keys:
            parts.append(f"{rot}: " + ", ".join(objects.CATEGORIES[k].label.lower() if k in objects.CATEGORIES else k
                                                 for k in keys))
    return (", " + "; ".join(parts)) if parts else ""


def describe_step(step: dict) -> str:
    """Texto curto para listas: 'Filtro — Vintage (80%)'."""
    try:
        s = normalize_step(step)
    except ValueError as e:
        return f"⚠ {step.get('op', '?')}: {e}"
    op = REGISTRY[s["op"]]
    k = s["op"]
    if k == "ajustes":
        parts = [f"{adjustments.ADJUSTMENT_LABELS[n]} {s[n]:+d}" for n in adjustments.ADJUSTMENT_KEYS if s[n]]
        return f"{op.label} — " + (", ".join(parts) if parts else "neutro")
    if k == "filtro":
        return f"{op.label} — {filters.filter_label(s['nome'])} ({s['intensidade']}%)"
    if k == "redimensionar":
        return f"{op.label} — lado maior {s['lado_maior']} px"
    if k == "recortar_proporcao":
        return f"{op.label} — {s['proporcao']} ({s['ancora']})"
    if k == "girar":
        return f"{op.label} — {s['angulo']}°"
    if k == "endireitar":
        return f"{op.label} — {s['angulo']:+.1f}°"
    if k == "espelhar":
        return f"{op.label} — {s['direcao']}"
    if k == "borrar_regiao":
        return f"{op.label} — {blur.BLUR_TYPES[s['tipo']]}, {blur.SHAPES[s['forma']]}"
    if k == "remover_fundo":
        dest = "transparente" if s["fundo"] == "transparente" else f"cor {s['cor']}"
        return f"{op.label} — {background.MODELS[s['modelo']][0]}, fundo {dest}{_objs_text(s)}"
    if k == "desfocar_fundo":
        return f"{op.label} — {background.MODELS[s['modelo']][0]}, intensidade {s['intensidade']}{_objs_text(s)}"
    if k == "desfocar_rostos":
        return f"{op.label} — {blur.BLUR_TYPES[s['tipo']]}, sensibilidade {s['sensibilidade']}"
    if k == "remover_marca_regiao":
        return f"{op.label} — {inpaint.METHODS[s['metodo']].split(' (')[0]}, {s['w'] * 100:.0f}% × {s['h'] * 100:.0f}%"
    if k == "remover_marca_modelo":
        return f"{op.label} — modelo '{s['nome']}', {inpaint.METHODS[s['metodo']].split(' (')[0]}"
    if k == "remover_marca_estimada":
        return f"{op.label} — modelo '{s['nome']}' ({s['forca']}%)"
    if k == "moldura":
        return f"{op.label} — {frames.frame_label(s['nome'])}"
    if k == "recortar_relativo":
        return f"{op.label} — {s['w'] * 100:.0f}% × {s['h'] * 100:.0f}%"
    return op.label


# ----------------------------------------------------------- operações base

_REL = dict(kind="float", minimum=0.0, maximum=1.0)

register(Operation(
    "ajustes", "Ajustes de luz e cor", adjustments.adjust,
    tuple(Param(k, adjustments.ADJUSTMENT_LABELS[k], "int", 0, *adjustments.adjustment_range(k))
          for k in adjustments.ADJUSTMENT_KEYS),
))

register(Operation(
    "filtro", "Filtro", filters.apply_filter,
    (
        Param("nome", "Filtro", "choice", "pb",
              choices=tuple((k, filters.filter_label(k)) for k in filters.filter_keys())),
        Param("intensidade", "Intensidade (%)", "int", 100, 0, 100),
    ),
))

register(Operation(
    "redimensionar", "Redimensionar", geometry.resize_long_side,
    (
        Param("lado_maior", "Lado maior (px)", "int", 2048, 16, 30000),
        Param("ampliar", "Permitir ampliar", "bool", False),
    ),
))

register(Operation(
    "recortar_proporcao", "Recortar na proporção", geometry.crop_to_ratio,
    (
        Param("proporcao", "Proporção", "choice", "1:1",
              choices=tuple((r, r) for r in geometry.RATIO_CHOICES)),
        Param("ancora", "Posição", "choice", "centro",
              choices=(("centro", "Centro"), ("topo", "Topo"), ("base", "Base"),
                       ("esquerda", "Esquerda"), ("direita", "Direita"))),
    ),
))

register(Operation(
    "recortar_relativo", "Recortar área", geometry.crop_relative,
    (
        Param("x", "X (0–1)", default=0.0, **_REL),
        Param("y", "Y (0–1)", default=0.0, **_REL),
        Param("w", "Largura (0–1)", default=1.0, **_REL),
        Param("h", "Altura (0–1)", default=1.0, **_REL),
    ),
))

register(Operation(
    "girar", "Girar", geometry.rotate_angle,
    (Param("angulo", "Ângulo (horário)", "choice", 90,
           choices=((90, "90°"), (180, "180°"), (270, "270°"))),),
))

register(Operation(
    "endireitar", "Endireitar", geometry.straighten,
    (Param("angulo", "Ângulo (°)", "float", 0.0, -45.0, 45.0),),
))

register(Operation(
    "espelhar", "Espelhar", geometry.flip,
    (Param("direcao", "Direção", "choice", "horizontal",
           choices=(("horizontal", "Horizontal"), ("vertical", "Vertical"))),),
))

register(Operation(
    "borrar_regiao", "Borrar região", blur.blur_region,
    (
        Param("x", "X (0–1)", default=0.8, **_REL),
        Param("y", "Y (0–1)", default=0.9, **_REL),
        Param("w", "Largura (0–1)", default=0.2, **_REL),
        Param("h", "Altura (0–1)", default=0.1, **_REL),
        Param("forma", "Forma", "choice", "retangulo", choices=tuple(blur.SHAPES.items())),
        Param("tipo", "Tipo", "choice", "gaussiano", choices=tuple(blur.BLUR_TYPES.items())),
        Param("intensidade", "Intensidade", "int", 50, 1, 100),
        Param("cor", "Cor da tarja", "color", "#000000"),
    ),
))

def _check_objects(step: dict) -> str | None:
    try:
        keep, drop = objects.parse_keys(step.get("manter", "")), objects.parse_keys(step.get("apagar", ""))
    except ValueError as e:
        return str(e)
    if set(keep) & set(drop):
        return "o mesmo tipo está em manter e em apagar"
    modelo = step.get("objetos_modelo", objects.DEFAULT_MODEL)
    if (keep or drop) and modelo in objects.MODELS and not objects.model_available(modelo):
        return "o modelo de objetos ainda não foi baixado (aba Fundo → Objetos e pessoas)"
    return None


register(Operation(
    "remover_fundo", "Remover fundo", background.remove_background,
    (
        Param("modelo", "Modelo", "choice", background.DEFAULT_MODEL,
              choices=tuple((k, v[1]) for k, v in background.MODELS.items())),
        Param("encolher", "Afinar (+) / engordar (−) borda", "int", 0, -100, 100),
        Param("suavizar", "Suavizar borda", "int", 20, 0, 100),
        Param("descontaminar", "Descontaminar cor da borda", "bool", True),
        Param("fundo", "Fundo", "choice", "transparente",
              choices=(("transparente", "Transparente (PNG/WEBP)"), ("cor", "Cor sólida"))),
        Param("cor", "Cor do fundo", "color", "#ffffff"),
        Param("manter", "Manter só estes tipos (vazio = objeto principal)", "str", ""),
        Param("apagar", "Apagar estes tipos", "str", ""),
        Param("objetos_modelo", "Modelo de objetos", "choice", objects.DEFAULT_MODEL,
              choices=tuple((k, v[2]) for k, v in objects.MODELS.items())),
    ),
    "Exige o modelo baixado (aba Fundo). Fundo transparente: salve em PNG ou WEBP. "
    "Tipos de objeto: "+", ".join(objects.category_keys())+".",
    check=_check_objects,
))

register(Operation(
    "desfocar_fundo", "Desfocar fundo", background.blur_background_auto,
    (
        Param("modelo", "Modelo", "choice", background.DEFAULT_MODEL,
              choices=tuple((k, v[1]) for k, v in background.MODELS.items())),
        Param("intensidade", "Intensidade do desfoque", "int", 50, 1, 100),
        Param("encolher", "Afinar (+) / engordar (−) borda", "int", 0, -100, 100),
        Param("suavizar", "Suavizar borda", "int", 20, 0, 100),
        Param("manter", "Manter só estes tipos (vazio = objeto principal)", "str", ""),
        Param("apagar", "Apagar estes tipos", "str", ""),
        Param("objetos_modelo", "Modelo de objetos", "choice", objects.DEFAULT_MODEL,
              choices=tuple((k, v[2]) for k, v in objects.MODELS.items())),
    ),
    "Exige o modelo baixado (aba Fundo). Tipos de objeto: "+", ".join(objects.category_keys())+".",
    check=_check_objects,
))

register(Operation(
    "desfocar_rostos", "Ocultar rostos", faces.blur_faces,
    (
        Param("sensibilidade", "Sensibilidade", "int", 60, 0, 100),
        Param("margem", "Margem extra (%)", "int", 30, 0, 200),
        Param("tipo", "Tipo", "choice", "mosaico", choices=tuple(blur.BLUR_TYPES.items())),
        Param("intensidade", "Intensidade", "int", 70, 1, 100),
        Param("cor", "Cor da tarja", "color", "#000000"),
    ),
    "Exige o modelo de rostos baixado (aba Ocultar rosto).",
))

register(Operation(
    "remover_marca_regiao", "Remover marca d'água (área)", inpaint.remove_region,
    (
        Param("x", "X (0–1)", default=0.7, **_REL),
        Param("y", "Y (0–1)", default=0.9, **_REL),
        Param("w", "Largura (0–1)", default=0.28, **_REL),
        Param("h", "Altura (0–1)", default=0.08, **_REL),
        Param("metodo", "Método", "choice", "telea", choices=tuple(inpaint.METHODS.items())),
        Param("expandir", "Expandir a máscara", "int", 30, 0, 100),
        Param("raio", "Raio do preenchimento", "int", 50, 1, 100),
        Param("grao", "Recolocar o grão da foto", "bool", True),
        Param("contexto", "Contexto da IA (LaMa)", "int", 40, 0, 100),
        Param("emenda", "Suavizar a emenda", "int", 20, 0, 100),
        Param("grao_forca", "Quantidade de grão (%)", "int", 100, 0, 200),
    ),
    "Para marcas em posição fixa. LaMa exige o modelo baixado (aba Marca d'água).",
))


def _check_template(kind: str):
    def check(step: dict) -> str | None:
        nome = step.get("nome", "")
        if not nome or not watermark.template_exists(nome, kind):
            return f"modelo de marca '{nome}' não existe (aba Marca d'água → salvar/aprender modelo)"
        return None
    return check


register(Operation(
    "remover_marca_modelo", "Remover marca d'água (modelo de máscara)", watermark.remove_with_template,
    (
        Param("nome", "Modelo", "marca_mascara", ""),
        Param("metodo", "Método", "choice", "telea", choices=tuple(inpaint.METHODS.items())),
        Param("expandir", "Expandir a máscara", "int", 30, 0, 100),
        Param("raio", "Raio do preenchimento", "int", 50, 1, 100),
        Param("grao", "Recolocar o grão da foto", "bool", True),
        Param("contexto", "Contexto da IA (LaMa)", "int", 40, 0, 100),
        Param("emenda", "Suavizar a emenda", "int", 20, 0, 100),
        Param("grao_forca", "Quantidade de grão (%)", "int", 100, 0, 200),
    ),
    "Máscara salva na aba Marca d'água, ancorada ao canto; serve em fotos de tamanhos diferentes.",
    _check_template(watermark.KIND_MASK),
))

register(Operation(
    "remover_marca_estimada", "Remover marca d'água translúcida (aprendida)", watermark.remove_estimated,
    (
        Param("nome", "Modelo", "marca_alpha", ""),
        Param("forca", "Força (%)", "int", 100, 0, 150),
    ),
    "Marca branca/preta translúcida repetida em várias fotos; exige a mesma proporção.",
    _check_template(watermark.KIND_ALPHA),
))


register(Operation(
    "moldura", "Moldura", frames.apply_frame,
    (
        Param("nome", "Moldura", "choice", "polaroid",
              choices=tuple((k, frames.frame_label(k)) for k in frames.frame_keys())),
        Param("espessura", "Espessura / intensidade (50 = padrão)", "int", 50, 0, 100),
        Param("cor_padrao", "Usar a cor própria da moldura", "bool", True),
        Param("cor", "Cor (se não usar a padrão)", "color", "#ffffff"),
        Param("texto", "Legenda / data / marca na borda", "str", ""),
    ),
    "Bordas e molduras de papel, cinema, filme e filmadora. O tamanho da foto aumenta nas molduras de borda.",
))
