"""Pilha de desfazer/refazer.

Cada estado guarda a imagem e, quando a edição é reproduzível em lote, o
`step` (dicionário de receita) que a gerou. Assim, "Salvar edições como
receita" é só ler os steps do histórico.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

_ids = itertools.count(1)


@dataclass
class HistoryEntry:
    label: str
    image: np.ndarray
    step: dict | None = None
    uid: int = field(default_factory=lambda: next(_ids))


class History:
    def __init__(self, max_states: int = 30, max_bytes: int = 1_500_000_000):
        self.max_states = max_states
        self.max_bytes = max_bytes
        self._entries: list[HistoryEntry] = []
        self._index = -1
        self._trimmed_steps: list[dict] = []
        self._trimmed_manual = 0

    # --------------------------------------------------------------- estado
    def reset(self, image: np.ndarray, label: str = "Abrir") -> None:
        self._entries = [HistoryEntry(label, image, None)]
        self._index = 0
        self._trimmed_steps = []
        self._trimmed_manual = 0

    @property
    def current(self) -> np.ndarray | None:
        return self._entries[self._index].image if self._entries else None

    @property
    def current_uid(self) -> int | None:
        return self._entries[self._index].uid if self._entries else None

    @property
    def original(self) -> np.ndarray | None:
        return self._entries[0].image if self._entries else None

    @property
    def is_empty(self) -> bool:
        return not self._entries

    def can_undo(self) -> bool:
        return self._index > 0

    def can_redo(self) -> bool:
        return 0 <= self._index < len(self._entries) - 1

    def undo_label(self) -> str:
        return self._entries[self._index].label if self.can_undo() else ""

    def redo_label(self) -> str:
        return self._entries[self._index + 1].label if self.can_redo() else ""

    # ------------------------------------------------------------- operações
    def push(self, image: np.ndarray, label: str, step: dict | None = None) -> None:
        if not self._entries:
            raise RuntimeError("Histórico vazio: chame reset() ao abrir a imagem.")
        del self._entries[self._index + 1:]
        self._entries.append(HistoryEntry(label, image, step))
        self._index = len(self._entries) - 1
        self._trim()

    def undo(self) -> np.ndarray | None:
        if self.can_undo():
            self._index -= 1
        return self.current

    def redo(self) -> np.ndarray | None:
        if self.can_redo():
            self._index += 1
        return self.current

    def steps(self) -> tuple[list[dict], int]:
        """Steps reproduzíveis até o estado atual e quantas edições manuais
        (sem step, ex.: pincel) ficaram de fora."""
        steps = [dict(s) for s in self._trimmed_steps]
        manual = self._trimmed_manual
        for e in self._entries[1:self._index + 1]:
            if e.step is None:
                manual += 1
            else:
                steps.append(dict(e.step))
        return steps, manual

    def _trim(self) -> None:
        """Descarta os estados intermediários mais antigos (o original é mantido)."""
        def total() -> int:
            return sum(e.image.nbytes for e in self._entries)

        while len(self._entries) > 2 and (
            len(self._entries) > self.max_states or total() > self.max_bytes
        ):
            if self._index <= 1:
                break
            # O estado 1 sai da pilha (não dá mais para desfazer até ele), mas
            # seu step é guardado, pois o efeito continua nas imagens seguintes.
            dropped = self._entries.pop(1)
            if dropped.step is None:
                self._trimmed_manual += 1
            else:
                self._trimmed_steps.append(dict(dropped.step))
            self._index -= 1
