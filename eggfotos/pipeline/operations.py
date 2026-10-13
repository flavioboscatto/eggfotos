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

from ..core import adjustments, background, blur, faces, filters, geometry


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
        return f"{op.label} — {background.MODELS[s['modelo']][0]}, fundo {dest}"
    if k == "desfocar_fundo":
        return f"{op.label} — {background.MODELS[s['modelo']][0]}, intensidade {s['intensidade']}"
    if k == "desfocar_rostos":
        return f"{op.label} — {blur.BLUR_TYPES[s['tipo']]}, sensibilidade {s['sensibilidade']}"
    if k == "recortar_relativo":
        return f"{op.label} — {s['w'] * 100:.0f}% × {s['h'] * 100:.0f}%"
    return op.label


# ----------------------------------------------------------- operações base

_REL = dict(kind="float", minimum=0.0, maximum=1.0)

register(Operation(
    "ajustes", "Ajustes de luz e cor", adjustments.adjust,
    tuple(Param(k, adjustments.ADJUSTMENT_LABELS[k], "int", 0, -100, 100)
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
    ),
    "Exige o modelo baixado (aba Fundo). Fundo transparente: salve em PNG ou WEBP.",
))

register(Operation(
    "desfocar_fundo", "Desfocar fundo", background.blur_background_auto,
    (
        Param("modelo", "Modelo", "choice", background.DEFAULT_MODEL,
              choices=tuple((k, v[1]) for k, v in background.MODELS.items())),
        Param("intensidade", "Intensidade do desfoque", "int", 50, 1, 100),
        Param("encolher", "Afinar (+) / engordar (−) borda", "int", 0, -100, 100),
        Param("suavizar", "Suavizar borda", "int", 20, 0, 100),
    ),
    "Exige o modelo baixado (aba Fundo).",
))

register(Operation(
    "desfocar_rostos", "Desfocar rostos", faces.blur_faces,
    (
        Param("sensibilidade", "Sensibilidade", "int", 60, 0, 100),
        Param("margem", "Margem extra (%)", "int", 30, 0, 200),
        Param("tipo", "Tipo", "choice", "mosaico", choices=tuple(blur.BLUR_TYPES.items())),
        Param("intensidade", "Intensidade", "int", 70, 1, 100),
        Param("cor", "Cor da tarja", "color", "#000000"),
    ),
    "Exige o modelo de rostos baixado (aba Rostos).",
))
