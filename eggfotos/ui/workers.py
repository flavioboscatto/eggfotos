"""Threads de trabalho. Regra: threads só trocam arrays NumPy e objetos
Python com a UI; QPixmap é criado apenas na thread principal."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from ..errors import friendly_error
from ..core.filters import apply_filter, filter_keys
from ..pipeline.recipe import Recipe
from ..pipeline.runner import BatchJob, BatchRunner, preview_recipe


class ThumbnailWorker(QThread):
    """Gera a miniatura de cada filtro para o painel de filtros."""
    thumbReady = pyqtSignal(int, str, object)  # geração, chave, array

    def __init__(self, source: np.ndarray, generation: int, parent=None):
        super().__init__(parent)
        self._source = source
        self._generation = generation
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        for key in filter_keys():
            if self._stop:
                return
            try:
                out = apply_filter(self._source, key, 100)
            except Exception:  # noqa: BLE001 — miniatura com erro fica em branco
                continue
            self.thumbReady.emit(self._generation, key, out)


class PreviewWorker(QThread):
    """Prévia de receita sobre algumas fotos de amostra."""
    ready = pyqtSignal(object)

    def __init__(self, paths: list[Path], recipe: Recipe, parent=None):
        super().__init__(parent)
        self._paths = paths
        self._recipe = recipe

    def run(self) -> None:
        self.ready.emit(preview_recipe(self._paths, self._recipe, max_side=520))


class BatchWorker(QThread):
    progress = pyqtSignal(int, int, object)  # feitos, total, ItemResult
    finishedReport = pyqtSignal(object)      # BatchReport
    failed = pyqtSignal(str)

    def __init__(self, job: BatchJob, parent=None):
        super().__init__(parent)
        self._job = job
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            runner = BatchRunner(
                self._job,
                on_progress=lambda d, t, r: self.progress.emit(d, t, r),
                should_cancel=lambda: self._cancel,
            )
            self.finishedReport.emit(runner.run())
        except Exception as e:  # noqa: BLE001
            self.failed.emit(friendly_error(e))


class TaskWorker(QThread):
    """Executa uma função qualquer (sem Qt) em segundo plano: resultado ou erro."""
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, func: Callable[[], object], parent=None):
        super().__init__(parent)
        self._func = func

    def run(self) -> None:
        try:
            self.done.emit(self._func())
        except Exception as e:  # noqa: BLE001
            self.failed.emit(friendly_error(e))
