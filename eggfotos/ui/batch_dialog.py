"""Diálogo de processamento em lote: pasta de entrada → receita → pasta de saída."""
from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QSplitter, QVBoxLayout,
    QWidget,
)

from .color_picker import choose_color, style_color_button
from ..pipeline.operations import REGISTRY, Operation, describe_step, get_operation, normalize_step
from ..pipeline.recipe import (
    OUTPUT_FORMATS, OutputSettings, Recipe, list_saved_recipes, recipes_dir, safe_filename,
)
from ..pipeline.runner import BatchJob, find_images, validate_job
from .qt_utils import debounce, fit_pixmap
from .workers import BatchWorker, PreviewWorker

PREVIEW_COUNT = 3


# ======================================================== editor de um passo

class StepEditor(QWidget):
    """Formulário gerado a partir dos Params da operação."""
    changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._form = QFormLayout(self)
        self._op: Operation | None = None
        self._widgets: dict[str, QWidget] = {}
        self._loading = False
        self._title = QLabel("Selecione um passo")
        self._title.setObjectName("hint")
        self._form.addRow(self._title)

    def set_step(self, step: dict | None) -> None:
        while self._form.rowCount() > 1:
            self._form.removeRow(1)
        self._widgets.clear()
        self._op = None
        if step is None:
            self._title.setText("Selecione um passo")
            return
        self._loading = True
        op = get_operation(step["op"])
        self._op = op
        self._title.setText(op.label)
        step = normalize_step(step)
        for p in op.params:
            w = self._make_widget(p, step[p.name])
            self._widgets[p.name] = w
            self._form.addRow(p.label, w)
        self._loading = False

    def _make_widget(self, p, value) -> QWidget:
        emit = lambda *_: (not self._loading) and self.changed.emit(self.step())  # noqa: E731
        if p.kind == "int":
            w = QSpinBox()
            w.setRange(int(p.minimum if p.minimum is not None else -10**9),
                       int(p.maximum if p.maximum is not None else 10**9))
            w.setValue(int(value))
            w.valueChanged.connect(emit)
        elif p.kind == "float":
            w = QDoubleSpinBox()
            w.setDecimals(3)
            w.setSingleStep(0.01 if (p.maximum or 0) <= 1 else 0.5)
            w.setRange(float(p.minimum if p.minimum is not None else -1e9),
                       float(p.maximum if p.maximum is not None else 1e9))
            w.setValue(float(value))
            w.valueChanged.connect(emit)
        elif p.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(emit)
        elif p.kind == "choice":
            w = QComboBox()
            for val, label in p.choices:
                w.addItem(label, val)
            idx = next((i for i, (val, _) in enumerate(p.choices) if str(val) == str(value)), 0)
            w.setCurrentIndex(idx)
            w.currentIndexChanged.connect(emit)
        elif p.kind == "color":
            w = QPushButton()
            w.setProperty("color", str(value))
            style_color_button(w, str(value))

            def pick(_=False, btn=w):
                c = choose_color(self, btn.property("color"))
                if c:
                    btn.setProperty("color", c)
                    style_color_button(btn, c)
                    emit()
            w.clicked.connect(pick)
        else:
            w = QLineEdit(str(value))
            w.textChanged.connect(emit)
        return w

    def step(self) -> dict:
        if self._op is None:
            return {}
        out = {"op": self._op.key}
        for p in self._op.params:
            w = self._widgets[p.name]
            if p.kind in ("int", "float"):
                out[p.name] = w.value()
            elif p.kind == "bool":
                out[p.name] = w.isChecked()
            elif p.kind == "choice":
                out[p.name] = w.currentData()
            elif p.kind == "color":
                out[p.name] = w.property("color")
            else:
                out[p.name] = w.text()
        return out


# ============================================================ diálogo de lote

class BatchDialog(QDialog):
    def __init__(self, parent=None, initial_steps: list[dict] | None = None,
                 manual_skipped: int = 0, start_dir: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("EggFotos — Processar pasta")
        self.resize(1050, 760)
        self._steps: list[dict] = [dict(s) for s in (initial_steps or [])]
        self._worker: BatchWorker | None = None
        self._preview_worker: PreviewWorker | None = None
        self._last_report = None

        root = QVBoxLayout(self)

        # ---------------------------------------------------------- pastas
        g_dirs = QGroupBox("1. Pastas")
        gd = QGridLayout(g_dirs)
        self.ed_in = QLineEdit(start_dir or "")
        self.ed_out = QLineEdit()
        b_in = QPushButton("Escolher…")
        b_out = QPushButton("Escolher…")
        self.chk_rec = QCheckBox("Incluir subpastas")
        self.lbl_count = QLabel("")
        self.lbl_count.setObjectName("hint")
        gd.addWidget(QLabel("Entrada"), 0, 0)
        gd.addWidget(self.ed_in, 0, 1)
        gd.addWidget(b_in, 0, 2)
        gd.addWidget(self.chk_rec, 1, 1)
        gd.addWidget(self.lbl_count, 1, 2)
        gd.addWidget(QLabel("Saída"), 2, 0)
        gd.addWidget(self.ed_out, 2, 1)
        gd.addWidget(b_out, 2, 2)
        root.addWidget(g_dirs)

        # ---------------------------------------------------------- receita
        g_rec = QGroupBox("2. Receita (passos aplicados em ordem)")
        gr = QHBoxLayout(g_rec)
        left = QVBoxLayout()
        top = QHBoxLayout()
        self.cmb_saved = QComboBox()
        self.cmb_saved.setMinimumWidth(180)
        b_load = QPushButton("Carregar")
        b_save = QPushButton("Salvar receita…")
        b_folder = QPushButton("Abrir pasta de receitas")
        top.addWidget(QLabel("Salvas:"))
        top.addWidget(self.cmb_saved, 1)
        top.addWidget(b_load)
        left.addLayout(top)
        self.lst_steps = QListWidget()
        left.addWidget(self.lst_steps, 1)
        add = QHBoxLayout()
        self.cmb_op = QComboBox()
        for op in REGISTRY.values():
            self.cmb_op.addItem(op.label, op.key)
        b_add = QPushButton("+ Adicionar")
        add.addWidget(self.cmb_op, 1)
        add.addWidget(b_add)
        left.addLayout(add)
        mv = QHBoxLayout()
        self.b_rm = QPushButton("Remover")
        self.b_up = QPushButton("↑")
        self.b_down = QPushButton("↓")
        mv.addWidget(self.b_rm)
        mv.addWidget(self.b_up)
        mv.addWidget(self.b_down)
        mv.addStretch()
        mv.addWidget(b_save)
        mv.addWidget(b_folder)
        left.addLayout(mv)
        if manual_skipped:
            hint = QLabel(f"{manual_skipped} edição(ões) manual(is) da foto aberta (ex.: pincel) "
                          "não entram na receita.")
            hint.setObjectName("hint")
            hint.setWordWrap(True)
            left.addWidget(hint)
        lw = QWidget()
        lw.setLayout(left)
        self.editor = StepEditor()
        g_param = QGroupBox("Parâmetros do passo")
        QVBoxLayout(g_param).addWidget(self.editor)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(lw)
        split.addWidget(g_param)
        split.setSizes([560, 420])
        gr.addWidget(split)
        root.addWidget(g_rec, 1)

        # ------------------------------------------------------------ saída
        g_out = QGroupBox("3. Arquivos de saída")
        go = QHBoxLayout(g_out)
        f1 = QFormLayout()
        self.cmb_fmt = QComboBox()
        for k, label in OUTPUT_FORMATS.items():
            self.cmb_fmt.addItem(label, k)
        self.sp_q = QSpinBox()
        self.sp_q.setRange(1, 100)
        self.sp_q.setValue(92)
        f1.addRow("Formato", self.cmb_fmt)
        f1.addRow("Qualidade (JPG/WEBP)", self.sp_q)
        f2 = QFormLayout()
        self.ed_suf = QLineEdit("_egg")
        self.sp_workers = QSpinBox()
        self.sp_workers.setRange(1, max(1, os.cpu_count() or 1))
        self.sp_workers.setValue(max(1, min(4, (os.cpu_count() or 2) - 1)))
        f2.addRow("Sufixo do nome", self.ed_suf)
        f2.addRow("Processos em paralelo", self.sp_workers)
        f3 = QVBoxLayout()
        self.chk_exif = QCheckBox("Manter EXIF (data, câmera, GPS)")
        self.chk_exif.setChecked(True)
        self.chk_skip = QCheckBox("Pular fotos já processadas")
        self.chk_skip.setChecked(True)
        f3.addWidget(self.chk_exif)
        f3.addWidget(self.chk_skip)
        go.addLayout(f1)
        go.addLayout(f2)
        go.addLayout(f3)
        root.addWidget(g_out)

        # ------------------------------------------------------------ prévia
        g_prev = QGroupBox("4. Prévia (antes → depois)")
        gp = QVBoxLayout(g_prev)
        self.prev_row = QHBoxLayout()
        self._prev_labels: list[tuple[QLabel, QLabel]] = []
        for _ in range(PREVIEW_COUNT):
            a, b = QLabel(), QLabel()
            for lb in (a, b):
                lb.setFixedSize(150, 110)
                lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lb.setStyleSheet("background:#1b1b1f; border-radius:4px;")
            self.prev_row.addWidget(a)
            arrow = QLabel("→")
            arrow.setObjectName("hint")
            self.prev_row.addWidget(arrow)
            self.prev_row.addWidget(b)
            self.prev_row.addSpacing(12)
            self._prev_labels.append((a, b))
        self.prev_row.addStretch()
        self.b_prev = QPushButton("Atualizar prévia")
        self.prev_row.addWidget(self.b_prev)
        gp.addLayout(self.prev_row)
        root.addWidget(g_prev)

        # ------------------------------------------------------- execução
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.lbl_status = QLabel("")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(90)
        root.addWidget(self.progress)
        root.addWidget(self.lbl_status)
        root.addWidget(self.log)
        bottom = QHBoxLayout()
        self.b_open_out = QPushButton("Abrir pasta de saída")
        self.b_open_out.setEnabled(False)
        self.b_close = QPushButton("Fechar")
        self.b_cancel = QPushButton("Cancelar")
        self.b_cancel.setEnabled(False)
        self.b_run = QPushButton("▶ Processar")
        self.b_run.setObjectName("primary")
        bottom.addWidget(self.b_open_out)
        bottom.addStretch()
        bottom.addWidget(self.b_close)
        bottom.addWidget(self.b_cancel)
        bottom.addWidget(self.b_run)
        root.addLayout(bottom)

        # ----------------------------------------------------------- sinais
        self._count_timer = debounce(self, 400, self._update_count)
        self._prev_timer = debounce(self, 600, self._start_preview)
        self.ed_out.textEdited.connect(lambda _: self.ed_out.setProperty("auto", False))
        b_in.clicked.connect(self._choose_in)
        b_out.clicked.connect(self._choose_out)
        self.ed_in.textChanged.connect(self._on_input_changed)
        self.chk_rec.toggled.connect(lambda _: self._count_timer.start())
        b_add.clicked.connect(self._add_step)
        self.b_rm.clicked.connect(self._remove_step)
        self.b_up.clicked.connect(lambda: self._move_step(-1))
        self.b_down.clicked.connect(lambda: self._move_step(1))
        self.lst_steps.currentRowChanged.connect(self._on_row)
        self.editor.changed.connect(self._on_step_edited)
        b_load.clicked.connect(self._load_selected)
        b_save.clicked.connect(self._save_recipe)
        b_folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(recipes_dir()))))
        self.b_prev.clicked.connect(self._start_preview)
        self.b_run.clicked.connect(self._run)
        self.b_cancel.clicked.connect(self._cancel)
        self.b_close.clicked.connect(self.close)
        self.b_open_out.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.ed_out.text())))

        self._refresh_saved()
        self._refresh_steps()
        if start_dir:
            self._on_input_changed(start_dir)

    # ============================================================ pastas
    def _choose_in(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Pasta com as fotos", self.ed_in.text())
        if d:
            self.ed_in.setText(d)

    def _choose_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Pasta de saída", self.ed_out.text() or self.ed_in.text())
        if d:
            self.ed_out.setText(d)
            self.ed_out.setProperty("auto", False)
            self._count_timer.start()

    def _on_input_changed(self, text: str) -> None:
        # sugere a saída automaticamente enquanto o usuário não escolher outra
        if text and (not self.ed_out.text() or self.ed_out.property("auto")):
            self.ed_out.setText(str(Path(text) / "EggFotos_saida"))
            self.ed_out.setProperty("auto", True)
        self._count_timer.start()

    def _files(self) -> list[Path]:
        entrada = Path(self.ed_in.text())
        saida = Path(self.ed_out.text()) if self.ed_out.text() else None
        return find_images(entrada, self.chk_rec.isChecked(), exclude=saida)

    def _update_count(self) -> None:
        if not Path(self.ed_in.text()).is_dir():
            self.lbl_count.setText("Pasta não encontrada")
            return
        n = len(self._files())
        self.lbl_count.setText(f"{n} foto(s) encontrada(s)")
        self._start_preview()

    # ============================================================ passos
    def _refresh_steps(self, select: int | None = None) -> None:
        self._prev_timer.start()
        self.lst_steps.blockSignals(True)
        self.lst_steps.clear()
        for i, s in enumerate(self._steps, 1):
            self.lst_steps.addItem(f"{i}. {describe_step(s)}")
        self.lst_steps.blockSignals(False)
        if select is not None and 0 <= select < len(self._steps):
            self.lst_steps.setCurrentRow(select)
        else:
            self.editor.set_step(None)
        has = self.lst_steps.currentRow() >= 0
        for b in (self.b_rm, self.b_up, self.b_down):
            b.setEnabled(has)

    def _on_row(self, row: int) -> None:
        self.editor.set_step(self._steps[row] if 0 <= row < len(self._steps) else None)
        for b in (self.b_rm, self.b_up, self.b_down):
            b.setEnabled(row >= 0)

    def _on_step_edited(self, step: dict) -> None:
        row = self.lst_steps.currentRow()
        if 0 <= row < len(self._steps):
            self._steps[row] = step
            self.lst_steps.item(row).setText(f"{row + 1}. {describe_step(step)}")
            self._prev_timer.start()

    def _add_step(self) -> None:
        op = get_operation(self.cmb_op.currentData())
        self._steps.append(op.new_step())
        self._refresh_steps(len(self._steps) - 1)

    def _remove_step(self) -> None:
        row = self.lst_steps.currentRow()
        if 0 <= row < len(self._steps):
            del self._steps[row]
            self._refresh_steps(min(row, len(self._steps) - 1))

    def _move_step(self, delta: int) -> None:
        row = self.lst_steps.currentRow()
        new = row + delta
        if 0 <= row < len(self._steps) and 0 <= new < len(self._steps):
            self._steps[row], self._steps[new] = self._steps[new], self._steps[row]
            self._refresh_steps(new)

    # =========================================================== receitas
    def _refresh_saved(self) -> None:
        self.cmb_saved.clear()
        for p in list_saved_recipes():
            self.cmb_saved.addItem(p.stem, str(p))

    def recipe(self) -> Recipe:
        return Recipe(
            nome=self.cmb_saved.currentText() or "Receita",
            passos=[dict(s) for s in self._steps],
            saida=OutputSettings(
                formato=self.cmb_fmt.currentData(),
                qualidade=self.sp_q.value(),
                sufixo=self.ed_suf.text(),
                manter_exif=self.chk_exif.isChecked(),
            ),
        )

    def _apply_recipe(self, r: Recipe) -> None:
        self._steps = [dict(s) for s in r.passos]
        self.cmb_fmt.setCurrentIndex(max(0, self.cmb_fmt.findData(r.saida.formato)))
        self.sp_q.setValue(int(r.saida.qualidade))
        self.ed_suf.setText(r.saida.sufixo)
        self.chk_exif.setChecked(r.saida.manter_exif)
        self._refresh_steps(0 if self._steps else None)

    def _load_selected(self) -> None:
        path = self.cmb_saved.currentData()
        if not path:
            return
        try:
            self._apply_recipe(Recipe.load(path))
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Receita", f"Não foi possível carregar:\n{e}")

    def _save_recipe(self) -> None:
        r = self.recipe()
        errors = r.validate()
        if errors:
            QMessageBox.warning(self, "Receita", "\n".join(errors))
            return
        nome, ok = QInputDialog.getText(self, "Salvar receita", "Nome da receita:",
                                        text=self.cmb_saved.currentText() or "Minha receita")
        if not ok or not nome.strip():
            return
        r.nome = nome.strip()
        path = recipes_dir() / f"{safe_filename(r.nome)}.json"
        if path.exists() and QMessageBox.question(
                self, "Salvar receita", f"Já existe '{r.nome}'. Substituir?") != QMessageBox.StandardButton.Yes:
            return
        r.save(path)
        self._refresh_saved()
        self.cmb_saved.setCurrentIndex(self.cmb_saved.findData(str(path)))

    # ============================================================= prévia
    def _start_preview(self) -> None:
        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._prev_timer.start()  # tenta de novo quando a prévia atual terminar
            return
        recipe = self.recipe()
        if recipe.validate() or not Path(self.ed_in.text()).is_dir():
            return
        files = self._files()
        if not files:
            return
        # amostra espalhada pela pasta (primeira, do meio, última)
        idx = sorted({0, len(files) // 2, len(files) - 1})[:PREVIEW_COUNT]
        self.b_prev.setEnabled(False)
        self._preview_worker = PreviewWorker([files[i] for i in idx], recipe, self)
        self._preview_worker.ready.connect(self._show_preview)
        self._preview_worker.finished.connect(lambda: self.b_prev.setEnabled(True))
        self._preview_worker.start()

    def _show_preview(self, results) -> None:
        for (la, lb) in self._prev_labels:
            la.clear()
            lb.clear()
        for (before, after), (la, lb) in zip(results, self._prev_labels):
            la.setPixmap(fit_pixmap(before, 140))
            if isinstance(after, str):
                lb.setText("Erro")
                lb.setToolTip(after)
            else:
                lb.setPixmap(fit_pixmap(after, 140))
                lb.setToolTip("")

    # =========================================================== execução
    def _job(self) -> BatchJob:
        return BatchJob(
            entrada=Path(self.ed_in.text()),
            saida=Path(self.ed_out.text()),
            recipe=self.recipe(),
            recursivo=self.chk_rec.isChecked(),
            pular_existentes=self.chk_skip.isChecked(),
            workers=self.sp_workers.value(),
        )

    def _run(self) -> None:
        if not self.ed_out.text().strip():
            QMessageBox.warning(self, "Lote", "Escolha a pasta de saída.")
            return
        job = self._job()
        errors = validate_job(job)
        if errors:
            QMessageBox.warning(self, "Lote", "\n".join(errors))
            return
        n = len(self._files())
        if n == 0:
            QMessageBox.information(self, "Lote", "Nenhuma foto encontrada na pasta de entrada.")
            return
        self.log.clear()
        self.progress.setRange(0, n)
        self.progress.setValue(0)
        self.lbl_status.setText(f"Processando {n} foto(s)…")
        self._set_running(True)
        self._worker = BatchWorker(job, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finishedReport.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.lbl_status.setText("Cancelando… aguardando as fotos em andamento.")
            self.b_cancel.setEnabled(False)

    def _set_running(self, running: bool) -> None:
        self.b_run.setEnabled(not running)
        self.b_cancel.setEnabled(running)
        self.b_close.setEnabled(not running)
        for w in (self.ed_in, self.ed_out, self.lst_steps, self.editor, self.cmb_fmt):
            w.setEnabled(not running)

    def _on_progress(self, done: int, total: int, result) -> None:
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)
        self.lbl_status.setText(f"{done} de {total} — {Path(result.origem).name}")
        if result.status == "erro":
            self.log.appendPlainText(f"ERRO  {Path(result.origem).name}: {result.mensagem}")

    def _on_done(self, report) -> None:
        self._set_running(False)
        self._last_report = report
        self.b_open_out.setEnabled(True)
        status = "Cancelado" if report.cancelado else "Concluído"
        self.lbl_status.setText(
            f"{status}: {report.ok} ok, {report.erros} erro(s), {report.pulados} pulada(s) "
            f"em {report.segundos:.1f} s."
        )
        self.log.appendPlainText(f"Log completo: {report.log_path}")

    def _on_failed(self, msg: str) -> None:
        self._set_running(False)
        QMessageBox.critical(self, "Lote", msg)

    def _can_close(self) -> bool:
        if self._worker is not None and self._worker.isRunning():
            if QMessageBox.question(self, "Lote", "O lote está em andamento. Cancelar e fechar?") \
                    != QMessageBox.StandardButton.Yes:
                return False
            self._worker.cancel()
            self._worker.wait()
        if self._preview_worker is not None:
            self._preview_worker.wait(3000)
        return True

    def reject(self):  # tecla Esc
        if self._can_close():
            super().reject()

    def closeEvent(self, event):
        if self._can_close():
            super().closeEvent(event)
        else:
            event.ignore()
