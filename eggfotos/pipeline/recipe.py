"""Receita = lista de steps + configurações de saída, salva em JSON."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .operations import apply_step, normalize_step

OUTPUT_FORMATS = {"mesmo": "Mesmo do original", "jpg": "JPG", "png": "PNG", "webp": "WEBP"}


def recipes_dir() -> Path:
    d = Path.home() / ".eggfotos" / "receitas"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class OutputSettings:
    formato: str = "mesmo"
    qualidade: int = 92
    sufixo: str = "_egg"
    manter_exif: bool = True

    def validate(self) -> None:
        if self.formato not in OUTPUT_FORMATS:
            raise ValueError(f"Formato de saída inválido: {self.formato}")
        if not 1 <= int(self.qualidade) <= 100:
            raise ValueError("Qualidade deve ficar entre 1 e 100.")
        if re.search(r'[\\/:*?"<>|]', self.sufixo):
            raise ValueError("O sufixo contém caracteres não permitidos em nomes de arquivo.")


@dataclass
class Recipe:
    nome: str = "Nova receita"
    passos: list[dict] = field(default_factory=list)
    saida: OutputSettings = field(default_factory=OutputSettings)
    versao: int = 1

    # ------------------------------------------------------------------ uso
    def validate(self) -> list[str]:
        """Lista de erros (vazia = receita válida)."""
        errors = []
        if not self.passos:
            errors.append("A receita não tem nenhum passo.")
        for i, step in enumerate(self.passos, 1):
            try:
                normalize_step(step)
            except ValueError as e:
                errors.append(f"Passo {i}: {e}")
        try:
            self.saida.validate()
        except ValueError as e:
            errors.append(str(e))
        return errors

    def apply(self, img: np.ndarray) -> np.ndarray:
        for step in self.passos:
            img = apply_step(img, step)
        return img

    # ---------------------------------------------------------- serialização
    def to_dict(self) -> dict:
        return {
            "nome": self.nome,
            "versao": self.versao,
            "passos": [normalize_step(s) for s in self.passos],
            "saida": asdict(self.saida),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Recipe":
        saida = OutputSettings(**{
            k: v for k, v in data.get("saida", {}).items()
            if k in OutputSettings.__dataclass_fields__
        })
        return cls(
            nome=data.get("nome", "Receita"),
            passos=[dict(s) for s in data.get("passos", [])],
            saida=saida,
            versao=int(data.get("versao", 1)),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Recipe":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def safe_filename(nome: str) -> str:
    s = re.sub(r'[\\/:*?"<>|]+', "_", nome).strip(" .")
    return s or "receita"


def list_saved_recipes() -> list[Path]:
    return sorted(recipes_dir().glob("*.json"), key=lambda p: p.stem.lower())
