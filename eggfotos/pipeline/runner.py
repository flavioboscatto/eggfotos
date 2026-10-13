"""Executor em lote: aplica uma receita a todas as fotos de uma pasta.

Não depende de Qt: pode ser usado por linha de comando ou pela interface
(que só recebe os callbacks de progresso).

Regras de segurança:
- os originais nunca são sobrescritos (saída sempre em outro caminho);
- a pasta de saída é ignorada na varredura, mesmo se estiver dentro da entrada;
- cada foto é independente: um erro não interrompe o lote, vai para o log CSV.
"""
from __future__ import annotations

import csv
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from ..core.geometry import downscale
from ..core.image_io import SUPPORTED_EXTENSIONS, load_image, save_image
from .recipe import Recipe


@dataclass
class ItemResult:
    origem: str
    destino: str
    status: str  # "ok" | "erro" | "pulado"
    mensagem: str = ""
    segundos: float = 0.0


@dataclass
class BatchJob:
    entrada: Path
    saida: Path
    recipe: Recipe
    recursivo: bool = False
    pular_existentes: bool = True
    workers: int = 0  # 0 = automático

    def resolved_workers(self) -> int:
        if self.workers and self.workers > 0:
            return self.workers
        # Fotos grandes ocupam muita memória por processo: limita a 4.
        return max(1, min(4, (os.cpu_count() or 2) - 1))


@dataclass
class BatchReport:
    total: int = 0
    ok: int = 0
    erros: int = 0
    pulados: int = 0
    cancelado: bool = False
    segundos: float = 0.0
    log_path: str = ""
    resultados: list[ItemResult] = field(default_factory=list)


def _same_or_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def find_images(folder: Path, recursive: bool = False, exclude: Path | None = None) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    it = folder.rglob("*") if recursive else folder.glob("*")
    files = []
    for p in it:
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if p.name.startswith("."):
            continue
        if exclude is not None and _same_or_inside(p, exclude):
            continue
        files.append(p)
    return sorted(files, key=lambda p: str(p).lower())


def output_path_for(src: Path, job: BatchJob) -> Path:
    """Mantém a estrutura de subpastas da entrada dentro da saída."""
    rel = src.relative_to(job.entrada)
    fmt = job.recipe.saida.formato
    ext = src.suffix.lower() if fmt == "mesmo" else f".{fmt}"
    if ext == ".jpeg":
        ext = ".jpg"
    name = f"{src.stem}{job.recipe.saida.sufixo}{ext}"
    return job.saida / rel.parent / name


def validate_job(job: BatchJob) -> list[str]:
    errors = list(job.recipe.validate())
    if not Path(job.entrada).is_dir():
        errors.append("A pasta de entrada não existe.")
    if Path(job.saida).resolve() == Path(job.entrada).resolve() and not job.recipe.saida.sufixo:
        errors.append(
            "Saída igual à entrada e sem sufixo: os originais seriam sobrescritos. "
            "Escolha outra pasta ou defina um sufixo."
        )
    return errors


def process_one(src: str, dst: str, recipe_dict: dict) -> ItemResult:
    """Processa uma foto. Função de nível de módulo para funcionar no
    ProcessPoolExecutor (precisa ser 'picklable' no Windows)."""
    t0 = time.perf_counter()
    try:
        recipe = Recipe.from_dict(recipe_dict)
        img, meta = load_image(src)
        out = recipe.apply(img)
        save_image(out, dst, quality=recipe.saida.qualidade, meta=meta,
                   keep_metadata=recipe.saida.manter_exif)
        return ItemResult(src, dst, "ok", "", time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001 — o erro vai para o log, o lote continua
        return ItemResult(src, dst, "erro", f"{type(e).__name__}: {e}", time.perf_counter() - t0)


ProgressCallback = Callable[[int, int, ItemResult], None]


class BatchRunner:
    def __init__(
        self,
        job: BatchJob,
        on_progress: ProgressCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ):
        self.job = job
        self.on_progress = on_progress or (lambda done, total, r: None)
        self.should_cancel = should_cancel or (lambda: False)

    def run(self) -> BatchReport:
        job = self.job
        job.entrada, job.saida = Path(job.entrada), Path(job.saida)
        errors = validate_job(job)
        if errors:
            raise ValueError("\n".join(errors))

        t0 = time.perf_counter()
        report = BatchReport()
        exclude = job.saida if job.saida.resolve() != job.entrada.resolve() else None
        files = find_images(job.entrada, job.recursivo, exclude=exclude)
        report.total = len(files)
        recipe_dict = job.recipe.to_dict()
        done = 0

        def record(r: ItemResult) -> None:
            nonlocal done
            done += 1
            report.resultados.append(r)
            if r.status == "ok":
                report.ok += 1
            elif r.status == "erro":
                report.erros += 1
            else:
                report.pulados += 1
            self.on_progress(done, report.total, r)

        tasks: list[tuple[Path, Path]] = []
        for src in files:
            dst = output_path_for(src, job)
            if job.pular_existentes and dst.exists():
                record(ItemResult(str(src), str(dst), "pulado", "já existia"))
            else:
                tasks.append((src, dst))

        workers = job.resolved_workers()
        if workers <= 1 or len(tasks) <= 1:
            for src, dst in tasks:
                if self.should_cancel():
                    report.cancelado = True
                    break
                record(process_one(str(src), str(dst), recipe_dict))
        else:
            self._run_parallel(tasks, recipe_dict, workers, record, report)

        report.segundos = time.perf_counter() - t0
        report.log_path = str(self._write_log(report))
        return report

    def _run_parallel(self, tasks, recipe_dict, workers, record, report) -> None:
        """Janela deslizante: no máximo 2×workers tarefas em voo, para o
        cancelamento responder rápido e a memória não explodir."""
        it = iter(tasks)
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                pending = set()

                def submit_next() -> bool:
                    try:
                        src, dst = next(it)
                    except StopIteration:
                        return False
                    pending.add(ex.submit(process_one, str(src), str(dst), recipe_dict))
                    return True

                for _ in range(workers * 2):
                    if not submit_next():
                        break
                while pending:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in finished:
                        record(f.result())
                    if self.should_cancel():
                        report.cancelado = True
                        continue  # deixa terminar o que já está em voo
                    while len(pending) < workers * 2 and submit_next():
                        pass
        except BrokenProcessPool:
            # Um processo morreu (ex.: falta de memória). Termina o resto em série.
            processed = {r.origem for r in report.resultados}
            remaining = [(s, d) for s, d in tasks if str(s) not in processed]
            for src, dst in remaining:
                if self.should_cancel():
                    report.cancelado = True
                    break
                record(process_one(str(src), str(dst), recipe_dict))

    def _write_log(self, report: BatchReport) -> Path:
        self.job.saida.mkdir(parents=True, exist_ok=True)
        path = self.job.saida / f"eggfotos_log_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig abre certo no Excel
            w = csv.writer(f, delimiter=";")
            w.writerow(["status", "origem", "destino", "segundos", "mensagem"])
            for r in report.resultados:
                w.writerow([r.status, r.origem, r.destino, f"{r.segundos:.2f}", r.mensagem])
        return path


def preview_recipe(paths: list[Path], recipe: Recipe, max_side: int = 640) -> list[tuple[np.ndarray, np.ndarray | str]]:
    """Prévia rápida: reduz cada foto ANTES de aplicar a receita.
    Retorna [(antes, depois_ou_mensagem_de_erro), ...]."""
    out = []
    for p in paths:
        try:
            img, _ = load_image(p)
            small, _ = downscale(img, max_side)
        except Exception as e:  # noqa: BLE001
            out.append((np.zeros((10, 10, 3), np.uint8), f"Erro ao abrir: {e}"))
            continue
        try:
            out.append((small, recipe.apply(small)))
        except Exception as e:  # noqa: BLE001
            out.append((small, f"{type(e).__name__}: {e}"))
    return out
