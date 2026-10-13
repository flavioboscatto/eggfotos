"""Painéis laterais: Ajustes, Filtros (com miniaturas), Recortar e girar, Borrar."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListView, QListWidget,
    QListWidgetItem, QPushButton, QRadioButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from ..core.adjustments import ADJUSTMENT_KEYS, ADJUSTMENT_LABELS, adjustment_range, neutral_params
from ..core.background import DEFAULT_MODEL, MODELS, model_available
from ..core import faces as faces_core
from ..core.blur import BLUR_TYPES
from ..core.filters import filter_keys, filter_label
from ..core.geometry import RATIO_CHOICES, downscale
from .color_picker import choose_color, style_color_button
from .qt_utils import debounce, to_qpixmap
from .workers import ThumbnailWorker


def _hint(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setObjectName("hint")
    lb.setWordWrap(True)
    return lb


def _slider(minimum: int, maximum: int, value: int) -> QSlider:
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(minimum, maximum)
    s.setValue(value)
    s.setPageStep(10)
    return s


# =================================================================== Ajustes

class AdjustPanel(QWidget):
    previewChanged = pyqtSignal(dict)
    applyRequested = pyqtSignal(dict)
    cancelRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        grid = QGridLayout()
        self._sliders: dict[str, QSlider] = {}
        self._values: dict[str, QLabel] = {}
        for row, key in enumerate(ADJUSTMENT_KEYS):
            s = _slider(*adjustment_range(key), 0)
            v = QLabel("0")
            v.setObjectName("value")
            v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            s.valueChanged.connect(lambda val, lb=v, k=key: lb.setText(
                f"{val:+d}" if val and adjustment_range(k)[0] < 0 else str(val)))
            s.valueChanged.connect(lambda _=0: self._timer.start())
            grid.addWidget(QLabel(ADJUSTMENT_LABELS[key]), row, 0)
            grid.addWidget(s, row, 1)
            grid.addWidget(v, row, 2)
            self._sliders[key] = s
            self._values[key] = v
        lay.addLayout(grid)
        lay.addWidget(_hint("A prévia é instantânea. Clique em Aplicar para gravar no histórico. "
            "Remover granulação: a prévia é aproximada, confira em 100% (Ctrl+1)."))

        btns = QHBoxLayout()
        self.btn_reset = QPushButton("Zerar")
        self.btn_apply = QPushButton("Aplicar")
        self.btn_apply.setObjectName("primary")
        btns.addWidget(self.btn_reset)
        btns.addStretch()
        btns.addWidget(self.btn_apply)
        lay.addLayout(btns)
        lay.addStretch()

        self._timer = debounce(self, 50, lambda: self.previewChanged.emit(self.values()))
        self.btn_apply.clicked.connect(lambda: self.applyRequested.emit(self.values()))
        self.btn_reset.clicked.connect(self._on_reset)

    def values(self) -> dict[str, int]:
        return {k: s.value() for k, s in self._sliders.items()}

    def reset(self) -> None:
        """Zera os controles sem emitir prévia."""
        self._timer.stop()
        for k, val in neutral_params().items():
            s = self._sliders[k]
            s.blockSignals(True)
            s.setValue(val)
            s.blockSignals(False)
            self._values[k].setText("0")

    def _on_reset(self) -> None:
        self.reset()
        self.cancelRequested.emit()


# =================================================================== Filtros

class FiltersPanel(QWidget):
    """Grade de miniaturas: cada filtro aparece aplicado à própria foto."""
    previewChanged = pyqtSignal(str, int)
    applyRequested = pyqtSignal(str, int)
    cancelRequested = pyqtSignal()

    THUMB = 104

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        self.list = QListWidget()
        self.list.setViewMode(QListView.ViewMode.IconMode)
        self.list.setIconSize(QSize(self.THUMB, self.THUMB))
        self.list.setGridSize(QSize(self.THUMB + 16, self.THUMB + 34))
        self.list.setResizeMode(QListView.ResizeMode.Adjust)
        self.list.setMovement(QListView.Movement.Static)
        self.list.setUniformItemSizes(True)
        self.list.setWordWrap(True)
        self.list.setSpacing(4)

        blank = QPixmap(self.THUMB, self.THUMB)
        blank.fill(QColor("#2d2d33"))
        self._items: dict[str, QListWidgetItem] = {}
        for key in filter_keys():
            item = QListWidgetItem(QIcon(blank), filter_label(key))
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.list.addItem(item)
            self._items[key] = item
        lay.addWidget(self.list, 1)

        row = QHBoxLayout()
        row.addWidget(QLabel("Intensidade"))
        self.slider = _slider(0, 100, 100)
        self.lbl_int = QLabel("100%")
        self.lbl_int.setObjectName("value")
        row.addWidget(self.slider, 1)
        row.addWidget(self.lbl_int)
        lay.addLayout(row)

        btns = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_apply = QPushButton("Aplicar filtro")
        self.btn_apply.setObjectName("primary")
        btns.addWidget(self.btn_cancel)
        btns.addStretch()
        btns.addWidget(self.btn_apply)
        lay.addLayout(btns)

        self._timer = debounce(self, 50, self._emit_preview)
        self.list.currentItemChanged.connect(lambda *_: self._emit_preview())
        self.list.itemClicked.connect(lambda *_: self._emit_preview())
        self.slider.valueChanged.connect(lambda v: (self.lbl_int.setText(f"{v}%"), self._timer.start()))
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_cancel.clicked.connect(self._on_cancel)

        self._worker: ThumbnailWorker | None = None
        self._generation = 0

    # -------------------------------------------------------------- seleção
    def selected_key(self) -> str | None:
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def reset(self) -> None:
        self._timer.stop()
        self.list.blockSignals(True)
        self.list.setCurrentRow(-1)
        self.list.clearSelection()
        self.list.blockSignals(False)
        self.slider.blockSignals(True)
        self.slider.setValue(100)
        self.slider.blockSignals(False)
        self.lbl_int.setText("100%")

    def _emit_preview(self) -> None:
        key = self.selected_key()
        if key:
            self.previewChanged.emit(key, self.slider.value())

    def _on_apply(self) -> None:
        key = self.selected_key()
        if key and key != "original":
            self.applyRequested.emit(key, self.slider.value())

    def _on_cancel(self) -> None:
        self.reset()
        self.cancelRequested.emit()

    # ----------------------------------------------------------- miniaturas
    def set_source(self, img: np.ndarray | None) -> None:
        """Regera as miniaturas a partir da foto atual (em segundo plano)."""
        self._stop_worker()
        self._generation += 1
        if img is None:
            return
        # recorte central quadrado: miniaturas ficam do mesmo tamanho na grade
        h, w = img.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        dpr = max(1.0, self.devicePixelRatioF())
        thumb, _ = downscale(img[y0:y0 + s, x0:x0 + s], int(self.THUMB * dpr))
        self._worker = ThumbnailWorker(np.ascontiguousarray(thumb), self._generation, self)
        self._worker.thumbReady.connect(self._on_thumb)
        self._worker.start()

    def _on_thumb(self, generation: int, key: str, arr: np.ndarray) -> None:
        if generation != self._generation or key not in self._items:
            return  # miniatura de uma foto anterior
        pm = to_qpixmap(arr)
        pm.setDevicePixelRatio(max(1.0, self.devicePixelRatioF()))
        self._items[key].setIcon(QIcon(pm))

    def _stop_worker(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(2000)
            self._worker = None

    def shutdown(self) -> None:
        self._stop_worker()


# ====================================================== Recortar e girar

class CropPanel(QWidget):
    ratioChanged = pyqtSignal(str)
    applyCrop = pyqtSignal()
    clearCrop = pyqtSignal()
    rotate90 = pyqtSignal(int)        # 90 ou 270
    flip = pyqtSignal(str)            # "horizontal" | "vertical"
    straightenPreview = pyqtSignal(float)
    straightenApply = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)

        g1 = QGroupBox("Recortar")
        f1 = QVBoxLayout(g1)
        form = QFormLayout()
        self.cmb_ratio = QComboBox()
        self.cmb_ratio.addItem("Livre", "livre")
        self.cmb_ratio.addItem("Original", "original")
        for r in RATIO_CHOICES:
            self.cmb_ratio.addItem(r, r)
        form.addRow("Proporção", self.cmb_ratio)
        f1.addLayout(form)
        self.lbl_sel = _hint("Arraste sobre a foto para selecionar. Arraste por dentro para mover.")
        f1.addWidget(self.lbl_sel)
        b = QHBoxLayout()
        self.btn_clear = QPushButton("Limpar")
        self.btn_crop = QPushButton("Recortar (Enter)")
        self.btn_crop.setObjectName("primary")
        self.btn_crop.setEnabled(False)
        b.addWidget(self.btn_clear)
        b.addStretch()
        b.addWidget(self.btn_crop)
        f1.addLayout(b)
        lay.addWidget(g1)

        g2 = QGroupBox("Girar e espelhar")
        f2 = QGridLayout(g2)
        btn_l = QPushButton("⟲ 90°")
        btn_r = QPushButton("⟳ 90°")
        btn_h = QPushButton("⇋ Horizontal")
        btn_v = QPushButton("⇵ Vertical")
        f2.addWidget(btn_l, 0, 0)
        f2.addWidget(btn_r, 0, 1)
        f2.addWidget(btn_h, 1, 0)
        f2.addWidget(btn_v, 1, 1)
        lay.addWidget(g2)

        g3 = QGroupBox("Endireitar")
        f3 = QVBoxLayout(g3)
        row = QHBoxLayout()
        self.sl_angle = _slider(-450, 450, 0)  # décimos de grau
        self.lbl_angle = QLabel("0,0°")
        self.lbl_angle.setObjectName("value")
        row.addWidget(self.sl_angle, 1)
        row.addWidget(self.lbl_angle)
        f3.addLayout(row)
        f3.addWidget(_hint("Gira e recorta as bordas vazias automaticamente."))
        b3 = QHBoxLayout()
        self.btn_angle_reset = QPushButton("Zerar")
        self.btn_angle_apply = QPushButton("Aplicar")
        self.btn_angle_apply.setObjectName("primary")
        b3.addWidget(self.btn_angle_reset)
        b3.addStretch()
        b3.addWidget(self.btn_angle_apply)
        f3.addLayout(b3)
        lay.addWidget(g3)
        lay.addStretch()

        self._timer = debounce(self, 60, lambda: self.straightenPreview.emit(self.angle()))
        self.cmb_ratio.currentIndexChanged.connect(lambda _: self.ratioChanged.emit(self.cmb_ratio.currentData()))
        self.btn_crop.clicked.connect(self.applyCrop)
        self.btn_clear.clicked.connect(self.clearCrop)
        btn_l.clicked.connect(lambda: self.rotate90.emit(270))
        btn_r.clicked.connect(lambda: self.rotate90.emit(90))
        btn_h.clicked.connect(lambda: self.flip.emit("horizontal"))
        btn_v.clicked.connect(lambda: self.flip.emit("vertical"))
        self.sl_angle.valueChanged.connect(self._on_angle)
        self.btn_angle_reset.clicked.connect(lambda: self.sl_angle.setValue(0))
        self.btn_angle_apply.clicked.connect(lambda: self.straightenApply.emit(self.angle()))

    def ratio_key(self) -> str:
        return self.cmb_ratio.currentData()

    def angle(self) -> float:
        return self.sl_angle.value() / 10.0

    def reset_angle(self) -> None:
        self._timer.stop()
        self.sl_angle.blockSignals(True)
        self.sl_angle.setValue(0)
        self.sl_angle.blockSignals(False)
        self.lbl_angle.setText("0,0°")

    def set_selection(self, rect) -> None:
        self.btn_crop.setEnabled(rect is not None)
        if rect is None:
            self.lbl_sel.setText("Arraste sobre a foto para selecionar. Arraste por dentro para mover.")
        else:
            self.lbl_sel.setText(f"Seleção: {rect[2]} × {rect[3]} px")

    def _on_angle(self, v: int) -> None:
        self.lbl_angle.setText(f"{v / 10:+.1f}°".replace(".", ","))
        self._timer.start()


# ==================================================================== Borrar

class BlurPanel(QWidget):
    settingsChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)

        g1 = QGroupBox("Ferramenta")
        f1 = QVBoxLayout(g1)
        self.grp_tool = QButtonGroup(self)
        for i, (key, text) in enumerate(
            (("pincel", "Pincel (pinte sobre a área)"),
             ("retangulo", "Retângulo (arraste)"),
             ("elipse", "Elipse (arraste)"))
        ):
            rb = QRadioButton(text)
            rb.setProperty("key", key)
            self.grp_tool.addButton(rb, i)
            f1.addWidget(rb)
        self.grp_tool.button(0).setChecked(True)
        lay.addWidget(g1)

        g2 = QGroupBox("Efeito")
        form = QFormLayout(g2)
        self.cmb_tipo = QComboBox()
        for k, label in BLUR_TYPES.items():
            self.cmb_tipo.addItem(label, k)
        form.addRow("Tipo", self.cmb_tipo)
        self.sl_int = _slider(1, 100, 50)
        form.addRow("Intensidade", self.sl_int)
        self.sp_brush = QSpinBox()
        self.sp_brush.setRange(2, 2000)
        self.sp_brush.setValue(80)
        self.sp_brush.setSuffix(" px")
        form.addRow("Tamanho do pincel", self.sp_brush)
        self.btn_color = QPushButton()
        self._color = "#000000"
        self._paint_color_button()
        form.addRow("Cor da tarja", self.btn_color)
        self.chk_original = QCheckBox("Usar a cor original da área (média)")
        self.chk_original.setToolTip("A tarja é preenchida com a cor média da própria região borrada")
        form.addRow(self.chk_original)
        lay.addWidget(g2)

        lay.addWidget(_hint(
            "Retângulo e elipse entram em 'Salvar edições como receita' e podem ir para o lote. "
            "O pincel é manual (só nesta foto). Segure Espaço para mover a foto."
        ))
        lay.addWidget(_hint(
            "Para anonimizar de verdade (rostos, placas), prefira Mosaico forte ou Tarja: "
            "desfoque leve pode ser parcialmente revertido."
        ))
        lay.addStretch()

        self.grp_tool.idToggled.connect(lambda _id, on: on and self._changed())
        self.cmb_tipo.currentIndexChanged.connect(lambda _: self._changed())
        self.chk_original.toggled.connect(lambda _: self._changed())
        self.sp_brush.valueChanged.connect(lambda _: self.settingsChanged.emit())
        self.btn_color.clicked.connect(self._pick_color)
        self._changed()

    def tool(self) -> str:
        return self.grp_tool.checkedButton().property("key")

    def tipo(self) -> str:
        return self.cmb_tipo.currentData()

    def intensidade(self) -> int:
        return self.sl_int.value()

    def brush_size(self) -> int:
        return self.sp_brush.value()

    def cor(self) -> str:
        return self._color

    def cor_original(self) -> bool:
        return self.chk_original.isChecked() and self.tipo() == "solido"

    def _changed(self) -> None:
        solid = self.tipo() == "solido"
        self.sp_brush.setEnabled(self.tool() == "pincel")
        self.btn_color.setEnabled(solid and not self.chk_original.isChecked())
        self.chk_original.setEnabled(solid)
        self.sl_int.setEnabled(not solid)
        self.settingsChanged.emit()

    def _paint_color_button(self) -> None:
        style_color_button(self.btn_color, self._color)

    def _pick_color(self) -> None:
        c = choose_color(self, self._color, "Cor da tarja")
        if c:
            self._color = c
            self._paint_color_button()
            self.cmb_tipo.setCurrentIndex(self.cmb_tipo.findData("solido"))  # escolher cor = tarja
            self.chk_original.setChecked(False)


# ==================================================================== Fundo

class BackgroundPanel(QWidget):
    """Remover fundo: baixar modelo → detectar (lento, 1×) → refinar ao vivo → aplicar."""
    downloadRequested = pyqtSignal(str)
    detectRequested = pyqtSignal(str)
    previewChanged = pyqtSignal(dict)
    applyRequested = pyqtSignal(dict)
    cancelRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)

        g1 = QGroupBox("1. Detectar o objeto")
        f1 = QVBoxLayout(g1)
        self.cmb_modelo = QComboBox()
        for k, (_, label) in MODELS.items():
            self.cmb_modelo.addItem(label, k)
        self.cmb_modelo.setCurrentIndex(self.cmb_modelo.findData(DEFAULT_MODEL))
        self.lbl_model = QLabel()
        self.lbl_model.setObjectName("hint")
        self.btn_download = QPushButton("Baixar modelo…")
        self.btn_detect = QPushButton("Detectar objeto")
        self.btn_detect.setObjectName("primary")
        for w in (self.cmb_modelo, self.lbl_model, self.btn_download, self.btn_detect):
            f1.addWidget(w)
        lay.addWidget(g1)

        self.g2 = QGroupBox("2. Refinar")
        f2 = QFormLayout(self.g2)
        self.sl_encolher = _slider(-100, 100, 0)
        self.sl_suavizar = _slider(0, 100, 20)
        f2.addRow("Afinar (+) / engordar (−)", self.sl_encolher)
        f2.addRow("Suavizar borda", self.sl_suavizar)
        self.chk_decont = QCheckBox("Descontaminar cor da borda")
        self.chk_decont.setChecked(True)
        f2.addRow(self.chk_decont)
        self.rb_transp = QRadioButton("Fundo transparente")
        self.rb_cor = QRadioButton("Cor sólida")
        self.rb_blur = QRadioButton("Desfocar fundo")
        self.rb_transp.setChecked(True)
        self.btn_color = QPushButton()
        self._color = "#ffffff"
        self._paint_color_button()
        row = QHBoxLayout()
        row.addWidget(self.rb_cor)
        row.addWidget(self.btn_color)
        self.sl_blur = _slider(1, 100, 50)
        self.sl_blur.setEnabled(False)
        f2.addRow(self.rb_transp)
        f2.addRow(row)
        f2.addRow(self.rb_blur)
        f2.addRow("Intensidade do desfoque", self.sl_blur)
        lay.addWidget(self.g2)

        lay.addWidget(_hint(
            "Fundo transparente só vale em PNG ou WEBP. Desfocar fundo mantém o objeto nítido. A detecção roda uma vez; "
            "os ajustes de borda são instantâneos. Entra em 'Salvar edições como receita' "
            "e no lote (o modelo precisa estar baixado)."
        ))

        btns = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_apply = QPushButton("Aplicar")
        self.btn_apply.setObjectName("primary")
        btns.addWidget(self.btn_cancel)
        btns.addStretch()
        btns.addWidget(self.btn_apply)
        lay.addLayout(btns)
        lay.addStretch()

        self._timer = debounce(self, 60, lambda: self.previewChanged.emit(self.values()))
        for sl in (self.sl_encolher, self.sl_suavizar):
            sl.valueChanged.connect(lambda _=0: self._timer.start())
        self.chk_decont.toggled.connect(lambda _: self._timer.start())
        self.rb_transp.toggled.connect(lambda _: self._timer.start())
        self.rb_blur.toggled.connect(lambda on: (self.sl_blur.setEnabled(on), self._timer.start()))
        self.sl_blur.valueChanged.connect(lambda _=0: self._timer.start())
        self.btn_color.clicked.connect(self._pick_color)
        self.cmb_modelo.currentIndexChanged.connect(lambda _: self.refresh_model_status())
        self.btn_download.clicked.connect(lambda: self.downloadRequested.emit(self.modelo()))
        self.btn_detect.clicked.connect(lambda: self.detectRequested.emit(self.modelo()))
        self.btn_apply.clicked.connect(lambda: self.applyRequested.emit(self.values()))
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.set_detected(False)
        self.refresh_model_status()

    def modelo(self) -> str:
        return self.cmb_modelo.currentData()

    def values(self) -> dict:
        """Estado do painel. `fundo` = transparente | cor | desfoque (a janela escolhe o step)."""
        return {
            "modelo": self.modelo(),
            "encolher": self.sl_encolher.value(),
            "suavizar": self.sl_suavizar.value(),
            "descontaminar": self.chk_decont.isChecked(),
            "fundo": ("transparente" if self.rb_transp.isChecked()
                      else "desfoque" if self.rb_blur.isChecked() else "cor"),
            "cor": self._color,
            "desfoque": self.sl_blur.value(),
        }

    def refresh_model_status(self) -> None:
        ok = model_available(self.modelo())
        self.lbl_model.setText("✓ Modelo baixado" if ok else "Modelo ainda não baixado")
        self.btn_download.setVisible(not ok)
        self.btn_detect.setEnabled(ok)

    def set_detected(self, detected: bool) -> None:
        self.g2.setEnabled(detected)
        self.btn_apply.setEnabled(detected)

    def reset(self) -> None:
        self._timer.stop()
        self.set_detected(False)

    def _on_cancel(self) -> None:
        self.reset()
        self.cancelRequested.emit()

    def _paint_color_button(self) -> None:
        style_color_button(self.btn_color, self._color)

    def _pick_color(self) -> None:
        c = choose_color(self, self._color, "Cor do fundo")
        if c:
            self._color = c
            self._paint_color_button()
            self.rb_cor.setChecked(True)
            self._timer.start()


# =================================================================== Rostos

class FacesPanel(QWidget):
    """Desfoque automático de rostos: baixar modelo → detectar → ajustar ao vivo → aplicar."""
    downloadRequested = pyqtSignal()
    detectRequested = pyqtSignal()
    previewChanged = pyqtSignal(dict)
    applyRequested = pyqtSignal(dict)
    cancelRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)

        g1 = QGroupBox("1. Detectar rostos")
        f1 = QVBoxLayout(g1)
        self.lbl_model = QLabel()
        self.lbl_model.setObjectName("hint")
        self.btn_download = QPushButton("Baixar modelo…")
        self.btn_detect = QPushButton("Detectar rostos")
        self.btn_detect.setObjectName("primary")
        self.lbl_count = QLabel("")
        for w in (self.lbl_model, self.btn_download, self.btn_detect, self.lbl_count):
            f1.addWidget(w)
        lay.addWidget(g1)

        self.g2 = QGroupBox("2. Desfoque")
        f2 = QFormLayout(self.g2)
        self.cmb_tipo = QComboBox()
        for k, label in BLUR_TYPES.items():
            self.cmb_tipo.addItem(label, k)
        self.cmb_tipo.setCurrentIndex(self.cmb_tipo.findData("mosaico"))
        self.sl_int = _slider(1, 100, 70)
        self.sl_sens = _slider(0, 100, 60)
        self.sl_margem = _slider(0, 200, 30)
        self.btn_color = QPushButton()
        self._color = "#000000"
        self._paint_color_button()
        f2.addRow("Tipo", self.cmb_tipo)
        f2.addRow("Intensidade", self.sl_int)
        f2.addRow("Sensibilidade", self.sl_sens)
        f2.addRow("Margem extra", self.sl_margem)
        f2.addRow("Cor da tarja", self.btn_color)
        self.chk_original = QCheckBox("Usar a cor original da área (média)")
        f2.addRow(self.chk_original)
        lay.addWidget(self.g2)
        lay.addWidget(_hint(
            "Aumente a sensibilidade se algum rosto escapar; reduza se aparecerem falsos positivos. "
            "Para anonimizar de verdade, use Mosaico forte ou Tarja. Entra na receita e no lote "
            "(cada foto do lote tem os rostos detectados sozinhos)."
        ))

        btns = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_apply = QPushButton("Aplicar")
        self.btn_apply.setObjectName("primary")
        btns.addWidget(self.btn_cancel)
        btns.addStretch()
        btns.addWidget(self.btn_apply)
        lay.addLayout(btns)
        lay.addStretch()

        self._timer = debounce(self, 60, lambda: self.previewChanged.emit(self.values()))
        for sl in (self.sl_int, self.sl_sens, self.sl_margem):
            sl.valueChanged.connect(lambda _=0: self._timer.start())
        self.cmb_tipo.currentIndexChanged.connect(lambda _: (self._sync(), self._timer.start()))
        self.chk_original.toggled.connect(lambda _: (self._sync(), self._timer.start()))
        self.btn_color.clicked.connect(self._pick_color)
        self.btn_download.clicked.connect(self.downloadRequested)
        self.btn_detect.clicked.connect(self.detectRequested)
        self.btn_apply.clicked.connect(lambda: self.applyRequested.emit(self.values()))
        self.btn_cancel.clicked.connect(self._on_cancel)
        self._sync()
        self.set_detected(False)
        self.refresh_model_status()

    def values(self) -> dict:
        """Parâmetros do step 'desfocar_rostos'."""
        return {
            "sensibilidade": self.sl_sens.value(),
            "margem": self.sl_margem.value(),
            "tipo": self.cmb_tipo.currentData(),
            "intensidade": self.sl_int.value(),
            "cor": self._color,
            "cor_original": self.chk_original.isChecked() and self.cmb_tipo.currentData() == "solido",
        }

    def _sync(self) -> None:
        solid = self.cmb_tipo.currentData() == "solido"
        self.btn_color.setEnabled(solid and not self.chk_original.isChecked())
        self.chk_original.setEnabled(solid)
        self.sl_int.setEnabled(not solid)

    def refresh_model_status(self) -> None:
        ok = faces_core.model_available()
        self.lbl_model.setText("✓ Modelo baixado" if ok else "Modelo ainda não baixado (~230 KB)")
        self.btn_download.setVisible(not ok)
        self.btn_detect.setEnabled(ok)

    def set_detected(self, detected: bool, count: int | None = None) -> None:
        self.g2.setEnabled(detected)
        self.btn_apply.setEnabled(detected)
        self.lbl_count.setText("" if count is None else
                               ("Nenhum rosto encontrado." if count == 0 else f"{count} possível(is) rosto(s)."))

    def reset(self) -> None:
        self._timer.stop()
        self.set_detected(False)

    def _on_cancel(self) -> None:
        self.reset()
        self.cancelRequested.emit()

    def _paint_color_button(self) -> None:
        style_color_button(self.btn_color, self._color)

    def _pick_color(self) -> None:
        c = choose_color(self, self._color, "Cor da tarja")
        if c:
            self._color = c
            self._paint_color_button()
            self.cmb_tipo.setCurrentIndex(self.cmb_tipo.findData("solido"))
            self.chk_original.setChecked(False)
            self._timer.start()


# ============================================================ Redimensionar

class ResizeDialog(QDialog):
    def __init__(self, width: int, height: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Redimensionar")
        self._ratio = width / height
        form = QFormLayout(self)
        self.sp_w = QSpinBox()
        self.sp_h = QSpinBox()
        for sp, v in ((self.sp_w, width), (self.sp_h, height)):
            sp.setRange(1, 30000)
            sp.setValue(v)
            sp.setSuffix(" px")
        self.chk_lock = QCheckBox("Manter proporção")
        self.chk_lock.setChecked(True)
        form.addRow("Largura", self.sp_w)
        form.addRow("Altura", self.sp_h)
        form.addRow("", self.chk_lock)
        form.addRow(_hint(f"Tamanho atual: {width} × {height} px"))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self._sync = False
        self.sp_w.valueChanged.connect(self._w_changed)
        self.sp_h.valueChanged.connect(self._h_changed)

    def _w_changed(self, v: int) -> None:
        if self.chk_lock.isChecked() and not self._sync:
            self._sync = True
            self.sp_h.setValue(max(1, round(v / self._ratio)))
            self._sync = False

    def _h_changed(self, v: int) -> None:
        if self.chk_lock.isChecked() and not self._sync:
            self._sync = True
            self.sp_w.setValue(max(1, round(v * self._ratio)))
            self._sync = False

    def size_values(self) -> tuple[int, int]:
        return self.sp_w.value(), self.sp_h.value()
