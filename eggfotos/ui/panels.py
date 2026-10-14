"""Painéis laterais: Ajustes, Filtros e Molduras (com miniaturas), Recortar, Borrar, Fundo, Rostos, Marca d'água."""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QInputDialog, QMessageBox,
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListView, QListWidget,
    QListWidgetItem, QPushButton, QRadioButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from ..core.adjustments import ADJUSTMENT_KEYS, ADJUSTMENT_LABELS, adjustment_range, neutral_params
from ..core.background import DEFAULT_MODEL, MODELS, model_available
from ..core import faces as faces_core
from ..core import inpaint as inpaint_core
from ..core import objects as objects_core
from ..core import presets
from ..core.inpaint import METHODS as INPAINT_METHODS
from ..core.inpaint import POLARITIES
from ..core.blur import BLUR_TYPES
from ..core.filters import filter_keys, filter_label
from ..core.frames import FRAMES, apply_frame, frame_keys
from ..core.geometry import RATIO_CHOICES, downscale
from .color_picker import choose_color, style_color_button
from .qt_utils import debounce, to_qpixmap
from .workers import ThumbnailWorker


def _hint(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setObjectName("hint")
    lb.setWordWrap(True)
    return lb


def _slider_row(minimum: int, maximum: int, value: int, suffix: str = "") -> tuple[QWidget, QSlider]:
    """Slider com o número ao lado (para o usuário ver o valor exato)."""
    s = _slider(minimum, maximum, value)
    lb = QLabel(f"{value}{suffix}")
    lb.setObjectName("value")
    lb.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    s.valueChanged.connect(lambda v: lb.setText(f"{v}{suffix}"))
    box = QWidget()
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(s, 1)
    row.addWidget(lb)
    return box, s


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
        int_row, self.sl_int = _slider_row(1, 100, 50)
        form.addRow("Intensidade", int_row)
        self.sp_brush = QSpinBox()
        self.sp_brush.setRange(2, 2000)
        self.sp_brush.setValue(80)
        self.sp_brush.setSuffix(" px")
        form.addRow("Tamanho do pincel", self.sp_brush)
        self.btn_color = QPushButton()
        self._color = "#000000"
        self._paint_color_button()
        form.addRow("Cor da tarja", self.btn_color)
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

    def _changed(self) -> None:
        solid = self.tipo() == "solido"
        self.sp_brush.setEnabled(self.tool() == "pincel")
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


# ==================================================================== Fundo

class BackgroundPanel(QWidget):
    """Remover fundo: baixar modelo → detectar (lento, 1×) → refinar ao vivo → aplicar."""
    downloadRequested = pyqtSignal(str)
    detectRequested = pyqtSignal(str)
    clickModeRequested = pyqtSignal()
    clickUse = pyqtSignal()
    clickClear = pyqtSignal()
    clickExit = pyqtSignal()
    objectsDownloadRequested = pyqtSignal(str)
    objectsScanRequested = pyqtSignal(str)
    objectsUseRequested = pyqtSignal(dict)
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
        self.btn_click = QPushButton("Ou selecione por cliques…")
        self.btn_click.setToolTip("Clique no objeto (MobileSAM). Bom quando a detecção automática erra.")
        f1.addWidget(self.btn_click)
        self.box_click = QWidget()
        bc = QVBoxLayout(self.box_click)
        bc.setContentsMargins(0, 0, 0, 0)
        bc.addWidget(_hint("Clique no objeto. Clique esquerdo = é objeto; Alt+clique ou botão direito = "
                           "não é objeto. Refine com mais cliques."))
        row_c = QHBoxLayout()
        self.btn_click_use = QPushButton("Usar seleção")
        self.btn_click_use.setObjectName("primary")
        self.btn_click_clear = QPushButton("Limpar cliques")
        self.btn_click_exit = QPushButton("Sair")
        for b in (self.btn_click_use, self.btn_click_clear, self.btn_click_exit):
            row_c.addWidget(b)
        bc.addLayout(row_c)
        self.box_click.setVisible(False)
        f1.addWidget(self.box_click)
        lay.addWidget(g1)

        self.g_obj = QGroupBox("Ou escolha por tipo de objeto")
        fo = QVBoxLayout(self.g_obj)
        fo.addWidget(_hint("Reconhece pessoas, carros, prédios, casas, barcos, animais, árvores, céu… "
                           "Para cada tipo: Manter (fica na foto), Apagar (vira fundo) ou Ignorar. "
                           "Havendo tipos em Manter, só eles ficam; sem nenhum, vale a detecção principal "
                           "(passo 1) menos o que estiver em Apagar."))
        self.cmb_omodelo = QComboBox()
        for k, (_, _, label) in objects_core.MODELS.items():
            self.cmb_omodelo.addItem(label, k)
        self.lbl_omodel = QLabel()
        self.lbl_omodel.setObjectName("hint")
        self.btn_odownload = QPushButton("Baixar modelo de objetos…")
        self.btn_oscan = QPushButton("Reconhecer objetos")
        self.btn_oscan.setObjectName("primary")
        for w in (self.cmb_omodelo, self.lbl_omodel, self.btn_odownload, self.btn_oscan):
            fo.addWidget(w)
        self.box_objs = QWidget()
        self._objs_form = QFormLayout(self.box_objs)
        self._objs_form.setContentsMargins(0, 0, 0, 0)
        self._obj_combos: dict[str, QComboBox] = {}
        self.box_objs.setVisible(False)
        fo.addWidget(self.box_objs)
        self.chk_orefine = QCheckBox("Refinar bordas de pessoas e veículos (MobileSAM)")
        self.chk_orefine.setChecked(True)
        self.chk_orefine.setToolTip("Usa o MobileSAM para afiar o contorno de pessoas, veículos, barcos e "
                                    "animais. Prédios, céu e árvores ficam com a máscara do modelo de objetos.")
        self.chk_orefine.setVisible(False)
        self.btn_ouse = QPushButton("Usar esta seleção")
        self.btn_ouse.setObjectName("primary")
        self.btn_ouse.setVisible(False)
        fo.addWidget(self.chk_orefine)
        fo.addWidget(self.btn_ouse)
        lay.addWidget(self.g_obj)

        self.g2 = QGroupBox("2. Refinar")
        f2 = QFormLayout(self.g2)
        enc_row, self.sl_encolher = _slider_row(-100, 100, 0)
        suav_row, self.sl_suavizar = _slider_row(0, 100, 20)
        f2.addRow("Afinar (+) / engordar (−)", enc_row)
        f2.addRow("Suavizar borda", suav_row)
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
        blur_row, self.sl_blur = _slider_row(1, 100, 50)
        self.sl_blur.setEnabled(False)
        f2.addRow(self.rb_transp)
        f2.addRow(row)
        f2.addRow(self.rb_blur)
        f2.addRow("Intensidade do desfoque", blur_row)
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
        self.cmb_omodelo.currentIndexChanged.connect(lambda _: self.refresh_objects_status())
        self.btn_odownload.clicked.connect(lambda: self.objectsDownloadRequested.emit(self.omodelo()))
        self.btn_oscan.clicked.connect(lambda: self.objectsScanRequested.emit(self.omodelo()))
        self.btn_ouse.clicked.connect(lambda: self.objectsUseRequested.emit(self.objects_values()))
        self.btn_click.clicked.connect(self.clickModeRequested)
        self.btn_click_use.clicked.connect(self.clickUse)
        self.btn_click_clear.clicked.connect(self.clickClear)
        self.btn_click_exit.clicked.connect(self.clickExit)
        self.btn_apply.clicked.connect(lambda: self.applyRequested.emit(self.values()))
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.set_detected(False)
        self.refresh_model_status()
        self.refresh_objects_status()

    def modelo(self) -> str:
        return self.cmb_modelo.currentData()

    # ------------------------------------------------ tipos de objeto
    def omodelo(self) -> str:
        return self.cmb_omodelo.currentData()

    def refresh_objects_status(self) -> None:
        ok = objects_core.model_available(self.omodelo())
        self.lbl_omodel.setText("✓ Modelo baixado" if ok else "Modelo ainda não baixado")
        self.btn_odownload.setVisible(not ok)
        self.btn_oscan.setEnabled(ok)

    def set_objects(self, items: list[tuple[str, float, int]] | None) -> None:
        """Mostra os tipos encontrados [(chave, fração, quantidade)]; None limpa a lista."""
        while self._objs_form.rowCount():
            self._objs_form.removeRow(0)
        self._obj_combos.clear()
        for key, frac, n in items or []:
            cmb = QComboBox()
            for data, label in (("ignorar", "Ignorar"), ("manter", "Manter"), ("apagar", "Apagar")):
                cmb.addItem(label, data)
            if key == "pessoas":
                cmb.setCurrentIndex(1)       # pessoas são o caso mais comum: já vêm marcadas para manter
            qtd = f"{n} " + ("objeto" if n == 1 else "objetos")
            self._objs_form.addRow(f"{objects_core.category_label(key)} · {frac * 100:.0f}% · {qtd}", cmb)
            self._obj_combos[key] = cmb
        has = bool(self._obj_combos)
        self.box_objs.setVisible(has)
        self.chk_orefine.setVisible(has)
        self.btn_ouse.setVisible(has)

    def clear_objects(self) -> None:
        self.set_objects(None)

    def objects_values(self) -> dict:
        pick = lambda what: ",".join(k for k, c in self._obj_combos.items() if c.currentData() == what)  # noqa: E731
        return {"manter": pick("manter"), "apagar": pick("apagar"), "objetos_modelo": self.omodelo(),
                "refinar": self.chk_orefine.isChecked()}

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

    def set_click_mode(self, on: bool) -> None:
        self.box_click.setVisible(on)
        self.btn_click.setVisible(not on)

    def reset(self) -> None:
        self._timer.stop()
        self.set_detected(False)
        self.set_click_mode(False)

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
    """Ocultar rosto (desfoque/mosaico/tarja automáticos): baixar modelo → detectar → ajustar ao vivo → aplicar."""
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

        self.g2 = QGroupBox("2. Como ocultar")
        f2 = QFormLayout(self.g2)
        self.cmb_tipo = QComboBox()
        for k, label in BLUR_TYPES.items():
            self.cmb_tipo.addItem(label, k)
        self.cmb_tipo.setCurrentIndex(self.cmb_tipo.findData("mosaico"))
        int_row, self.sl_int = _slider_row(1, 100, 70)
        sens_row, self.sl_sens = _slider_row(0, 100, 60)
        margem_row, self.sl_margem = _slider_row(0, 200, 30)
        self.btn_color = QPushButton()
        self._color = "#000000"
        self._paint_color_button()
        f2.addRow("Tipo", self.cmb_tipo)
        f2.addRow("Intensidade", int_row)
        f2.addRow("Sensibilidade", sens_row)
        f2.addRow("Margem extra", margem_row)
        f2.addRow("Cor da tarja", self.btn_color)
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
        }

    def _sync(self) -> None:
        solid = self.cmb_tipo.currentData() == "solido"
        self.btn_color.setEnabled(solid)
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
            self._timer.start()


# ============================================================== Marca d'água

class WatermarkPanel(QWidget):
    """Marcar a área (pincel/retângulo/clique) → escolher método → remover."""
    settingsChanged = pyqtSignal()
    clearRequested = pyqtSignal()
    applyRequested = pyqtSignal(dict)
    downloadRequested = pyqtSignal()
    saveTemplateRequested = pyqtSignal()
    learnRequested = pyqtSignal()
    snapRequested = pyqtSignal(dict)
    snapUndoRequested = pyqtSignal()
    findRequested = pyqtSignal()
    suggestRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)

        g0 = QGroupBox("Ajuda automática (sem IA)")
        f0 = QVBoxLayout(g0)
        self.btn_find = QPushButton("Procurar a marca na foto")
        self.btn_find.setToolTip("Procura regiões com traços finos e coerentes (letras, logotipos), de "
                                 "preferência perto das bordas, e já as marca. É um palpite: confira.")
        self.btn_suggest = QPushButton("Sugerir ajustes para a área marcada")
        self.btn_suggest.setToolTip("Mede o fundo em volta da marcação (textura, grão, rostos por perto) e "
                                    "preenche os controles da seção 2.")
        self.lbl_sugestao = QLabel("")
        self.lbl_sugestao.setObjectName("hint")
        self.lbl_sugestao.setWordWrap(True)
        self.lbl_sugestao.setVisible(False)
        for w in (self.btn_find, self.btn_suggest, self.lbl_sugestao):
            f0.addWidget(w)
        lay.addWidget(g0)

        g1 = QGroupBox("1. Marque a área a apagar")
        f1 = QVBoxLayout(g1)
        self.grp_tool = QButtonGroup(self)
        for i, (key, text, tip) in enumerate((
                ("pincel", "Pincel", "Pinte sobre a marca d'água."),
                ("retangulo", "Retângulo", "Arraste um retângulo em volta da marca."),
                ("clique", "Clique no objeto (MobileSAM)", "Clique no objeto; Alt+clique ou botão direito tira da marcação."))):
            rb = QRadioButton(text)
            rb.setToolTip(tip)
            rb.setProperty("key", key)
            self.grp_tool.addButton(rb, i)
            f1.addWidget(rb)
        self.grp_tool.button(0).setChecked(True)
        self.sp_brush = QSpinBox()
        self.sp_brush.setRange(2, 2000)
        self.sp_brush.setValue(40)
        self.sp_brush.setSuffix(" px")
        form1 = QFormLayout()
        form1.addRow("Tamanho do pincel", self.sp_brush)
        f1.addLayout(form1)
        self.lbl_marked = QLabel("Nada marcado ainda.")
        self.lbl_marked.setObjectName("hint")
        self.btn_clear = QPushButton("Limpar marcação")
        f1.addWidget(self.lbl_marked)
        f1.addWidget(self.btn_clear)
        lay.addWidget(g1)

        g1b = QGroupBox("Ajuste aos traços da marca")
        f1b = QFormLayout(g1b)
        self.cmb_pol = QComboBox()
        for k, label in POLARITIES.items():
            self.cmb_pol.addItem(label, k)
        sens_row, self.sl_sens = _slider_row(0, 100, 40)
        self.btn_snap = QPushButton("Apertar marcação aos traços")
        self.btn_snap.setToolTip("Marque a área de forma solta (retângulo ou pincel largo) e clique aqui: "
                                 "a marcação encolhe para cobrir só as letras/traços da marca.")
        self.btn_snap_undo = QPushButton("Voltar à marcação anterior")
        f1b.addRow("A marca é", self.cmb_pol)
        f1b.addRow("Sensibilidade", sens_row)
        f1b.addRow(self.btn_snap)
        f1b.addRow(self.btn_snap_undo)
        lay.addWidget(g1b)

        g2 = QGroupBox("2. Como preencher")
        f2 = QFormLayout(g2)
        self.cmb_perfil = QComboBox()
        self.lbl_dica = QLabel("")
        self.lbl_dica.setObjectName("hint")
        self.lbl_dica.setWordWrap(True)
        self.btn_perfil_salvar = QPushButton("Salvar como perfil…")
        self.btn_perfil_excluir = QPushButton("Excluir perfil")
        row_perfil = QWidget()
        rp = QHBoxLayout(row_perfil)
        rp.setContentsMargins(0, 0, 0, 0)
        rp.addWidget(self.btn_perfil_salvar)
        rp.addWidget(self.btn_perfil_excluir)
        self.cmb_metodo = QComboBox()
        for k, label in INPAINT_METHODS.items():
            self.cmb_metodo.addItem(label, k)
        self.lbl_model = QLabel()
        self.lbl_model.setObjectName("hint")
        self.btn_download = QPushButton("Baixar modelo LaMa (~200 MB)…")
        exp_row, self.sl_expandir = _slider_row(0, 100, 30)
        raio_row, self.sl_raio = _slider_row(1, 100, 50)
        ctx_row, self.sl_contexto = _slider_row(0, 100, 40)
        emenda_row, self.sl_emenda = _slider_row(0, 100, 20)
        grao_row, self.sl_grao = _slider_row(0, 200, 100, "%")
        self.sl_contexto.setToolTip("LaMa: quanto da vizinhança a IA enxerga. Mais contexto = emenda mais "
                                    "coerente, porém mais janelas (mais lento).")
        self.sl_raio.setToolTip("Telea / Navier-Stokes: alcance do preenchimento a partir das bordas.")
        self.sl_emenda.setToolTip("Suavidade da borda onde o trecho preenchido encontra a foto.")
        self.sl_grao.setToolTip("Quanto do grão da foto ao redor é recolocado no trecho preenchido (0 = nenhum).")
        f2.addRow("Perfil", self.cmb_perfil)
        f2.addRow(self.lbl_dica)
        f2.addRow("Método", self.cmb_metodo)
        f2.addRow(self.lbl_model)
        f2.addRow(self.btn_download)
        f2.addRow("Expandir a marcação", exp_row)
        self.lbl_raio = QLabel("Raio (rápidos)")
        self.lbl_ctx = QLabel("Contexto (LaMa)")
        f2.addRow(self.lbl_raio, raio_row)
        f2.addRow(self.lbl_ctx, ctx_row)
        f2.addRow("Suavizar a emenda", emenda_row)
        f2.addRow("Grão da foto", grao_row)
        f2.addRow(row_perfil)
        lay.addWidget(g2)

        btns = QHBoxLayout()
        self.btn_apply = QPushButton("Remover marca d'água")
        self.btn_apply.setObjectName("primary")
        btns.addWidget(self.btn_apply)
        lay.addLayout(btns)

        g3 = QGroupBox("3. Para várias fotos (lote)")
        f3 = QVBoxLayout(g3)
        self.btn_save_tpl = QPushButton("Salvar marcação como modelo…")
        self.btn_save_tpl.setToolTip("Guarda a marcação ancorada ao canto, com tamanho proporcional à largura. "
                                     "No lote: passo \"Remover marca d'água (modelo de máscara)\".")
        self.lbl_tpl = QLabel("")
        self.lbl_tpl.setObjectName("hint")
        self.btn_learn = QPushButton("Aprender marca de uma pasta…")
        self.btn_learn.setToolTip("Para marca branca/preta translúcida na mesma posição em muitas fotos: "
                                  "estima a opacidade e desfaz sem inventar textura.")
        for w in (self.btn_save_tpl, self.lbl_tpl, self.btn_learn):
            f3.addWidget(w)
        lay.addWidget(g3)

        lay.addWidget(_hint(
            "Marque um pouco além da marca; \"Expandir\" cobre as bordas suaves. Rápido funciona bem em "
            "fundos lisos; LaMa é melhor em texturas e áreas grandes. Um único retângulo ou um modelo salvo "
            "entra na receita e no lote. Use só em fotos suas ou com autorização: remover marca d'água de "
            "imagens de terceiros pode violar direitos autorais."
        ))
        lay.addStretch()

        self._applying = False
        self.reload_profiles()
        self.cmb_perfil.currentIndexChanged.connect(self._on_profile_chosen)
        for sl in (self.sl_expandir, self.sl_raio, self.sl_contexto, self.sl_emenda, self.sl_grao, self.sl_sens):
            sl.valueChanged.connect(lambda _=0: self._mark_custom())
        self.cmb_metodo.currentIndexChanged.connect(lambda _: self._mark_custom())
        self.btn_perfil_salvar.clicked.connect(self._save_profile)
        self.btn_perfil_excluir.clicked.connect(self._delete_profile)
        self.grp_tool.idToggled.connect(lambda _id, on: on and self.settingsChanged.emit())
        self.sp_brush.valueChanged.connect(lambda _: self.settingsChanged.emit())
        self.cmb_metodo.currentIndexChanged.connect(lambda _: self.refresh_model_status())
        self.btn_clear.clicked.connect(self.clearRequested)
        self.btn_save_tpl.clicked.connect(self.saveTemplateRequested)
        self.btn_snap.clicked.connect(lambda: self.snapRequested.emit(
            {"polaridade": self.cmb_pol.currentData(), "sensibilidade": self.sl_sens.value()}))
        self.btn_snap_undo.clicked.connect(self.snapUndoRequested)
        self.btn_find.clicked.connect(self.findRequested)
        self.btn_suggest.clicked.connect(self.suggestRequested)
        self.btn_learn.clicked.connect(self.learnRequested)
        self.btn_download.clicked.connect(self.downloadRequested)
        self.btn_apply.clicked.connect(lambda: self.applyRequested.emit(self.values()))
        self.set_marked(0)
        self.set_snap_undo(False)
        self.refresh_model_status()

    # ------------------------------------------------------------- perfis
    def reload_profiles(self, select: str | None = None) -> None:
        """Recarrega a lista: Personalizado, perfis prontos e os do usuário."""
        self._applying = True
        self.cmb_perfil.clear()
        self.cmb_perfil.addItem(presets.CUSTOM, None)
        for key in presets.preset_keys():
            self.cmb_perfil.addItem(presets.PRESETS[key].label, f"preset:{key}")
        for nome in presets.load_user_profiles():
            self.cmb_perfil.addItem(f"★ {nome}", f"user:{nome}")
        i = self.cmb_perfil.findData(select) if select else 0
        self.cmb_perfil.setCurrentIndex(max(0, i))
        self._applying = False
        self._update_profile_ui()

    def _profile_values(self, data: str | None) -> tuple[dict, str] | None:
        if not data:
            return None
        kind, _, key = data.partition(":")
        if kind == "preset" and key in presets.PRESETS:
            p = presets.PRESETS[key]
            return p.values(), p.dica
        if kind == "user":
            v = presets.load_user_profiles().get(key)
            if v is not None:
                return v, "Perfil seu. Ajuste os controles e salve de novo com o mesmo nome para atualizar."
        return None

    def _on_profile_chosen(self) -> None:
        if self._applying:
            return
        got = self._profile_values(self.cmb_perfil.currentData())
        if got is not None:
            values, _ = got
            self._applying = True
            self.cmb_metodo.setCurrentIndex(max(0, self.cmb_metodo.findData(values["metodo"])))
            for sl, key in ((self.sl_expandir, "expandir"), (self.sl_raio, "raio"),
                            (self.sl_contexto, "contexto"), (self.sl_emenda, "emenda"),
                            (self.sl_grao, "grao_forca"), (self.sl_sens, "sensibilidade")):
                sl.setValue(int(values[key]))
            self._applying = False
            self.refresh_model_status()
        self._update_profile_ui()

    def _mark_custom(self) -> None:
        """Mexeu num controle: o conjunto deixa de ser o perfil escolhido."""
        if self._applying or self.cmb_perfil.currentIndex() == 0:
            return
        self._applying = True
        self.cmb_perfil.setCurrentIndex(0)
        self._applying = False
        self._update_profile_ui()

    def _update_profile_ui(self) -> None:
        data = self.cmb_perfil.currentData()
        got = self._profile_values(data)
        self.lbl_dica.setText(got[1] if got else "Ajuste os controles abaixo ou escolha um perfil pronto.")
        self.btn_perfil_excluir.setEnabled(bool(data) and data.startswith("user:"))

    def current_settings(self) -> dict:
        """Todos os controles do preenchimento (usado para salvar um perfil)."""
        v = self.values()
        return {"metodo": v["metodo"], "expandir": v["expandir"], "raio": v["raio"],
                "contexto": v["contexto"], "emenda": v["emenda"], "grao_forca": v["grao_forca"],
                "sensibilidade": self.sl_sens.value()}

    def _save_profile(self) -> None:
        nome, ok = QInputDialog.getText(self, "Salvar perfil", "Nome do perfil:", text="Meu perfil")
        if not ok or not nome.strip():
            return
        try:
            nome = presets.save_user_profile(nome, self.current_settings())
        except ValueError as e:
            QMessageBox.warning(self, "Perfil", str(e))
            return
        self.reload_profiles(select=f"user:{nome}")

    def _delete_profile(self) -> None:
        data = self.cmb_perfil.currentData() or ""
        if not data.startswith("user:"):
            return
        nome = data[5:]
        r = QMessageBox.question(self, "Excluir perfil", f"Excluir o perfil \"{nome}\"?")
        if r == QMessageBox.StandardButton.Yes:
            presets.delete_user_profile(nome)
            self.reload_profiles()

    def tool(self) -> str:
        return self.grp_tool.checkedButton().property("key")

    def brush_size(self) -> int:
        return self.sp_brush.value()

    def metodo(self) -> str:
        return self.cmb_metodo.currentData()

    def values(self) -> dict:
        """Parâmetros comuns do step 'remover_marca_regiao' (sem o retângulo)."""
        return {"metodo": self.metodo(), "expandir": self.sl_expandir.value(),
                "raio": self.sl_raio.value(), "contexto": self.sl_contexto.value(),
                "emenda": self.sl_emenda.value(),
                "grao": self.sl_grao.value() > 0, "grao_forca": self.sl_grao.value()}

    def refresh_model_status(self) -> None:
        lama = self.metodo() == "lama"
        ok = inpaint_core.lama_available()
        self.lbl_model.setVisible(lama)
        self.lbl_model.setText("✓ Modelo LaMa baixado" if ok else "Modelo LaMa ainda não baixado")
        self.btn_download.setVisible(lama and not ok)
        # cada método mostra só os controles que valem para ele
        self.lbl_raio.setVisible(not lama)
        self.sl_raio.parentWidget().setVisible(not lama)
        self.lbl_ctx.setVisible(lama)
        self.sl_contexto.parentWidget().setVisible(lama)

    def set_marked(self, count: int) -> None:
        self.lbl_marked.setText("Nada marcado ainda." if count == 0
                                else f"{count} marcação(ões) feita(s).")
        self.btn_apply.setEnabled(count > 0)
        self.btn_clear.setEnabled(count > 0)
        self.btn_save_tpl.setEnabled(count > 0)
        self.btn_snap.setEnabled(count > 0)
        self.btn_suggest.setEnabled(count > 0)

    def apply_suggestion(self, sug) -> None:
        """Preenche os controles com a sugestão da pré-análise e mostra o porquê."""
        v = sug.values
        self.cmb_metodo.setCurrentIndex(max(0, self.cmb_metodo.findData(v["metodo"])))
        for sl, key in ((self.sl_expandir, "expandir"), (self.sl_raio, "raio"),
                        (self.sl_contexto, "contexto"), (self.sl_emenda, "emenda"),
                        (self.sl_grao, "grao_forca"), (self.sl_sens, "sensibilidade")):
            sl.setValue(int(v[key]))
        self.cmb_pol.setCurrentIndex(max(0, self.cmb_pol.findData(sug.polaridade)))
        self.refresh_model_status()
        self.lbl_sugestao.setText("Sugestão aplicada:\n" + sug.resumo())
        self.lbl_sugestao.setVisible(True)

    def set_snap_undo(self, available: bool) -> None:
        self.btn_snap_undo.setEnabled(available)

    def set_template_name(self, nome: str | None) -> None:
        self.lbl_tpl.setText(f"Modelo salvo: {nome}" if nome else "")


# ================================================================== Molduras

class FramesPanel(QWidget):
    """Molduras com miniatura da própria foto, espessura, cor e legenda (prévia ao vivo)."""
    previewChanged = pyqtSignal(dict)
    applyRequested = pyqtSignal(dict)
    cancelRequested = pyqtSignal()

    THUMB = 104

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        self.list = QListWidget()
        self.list.setViewMode(QListView.ViewMode.IconMode)
        self.list.setIconSize(QSize(self.THUMB, self.THUMB))
        self.list.setGridSize(QSize(self.THUMB + 16, self.THUMB + 40))
        self.list.setResizeMode(QListView.ResizeMode.Adjust)
        self.list.setMovement(QListView.Movement.Static)
        self.list.setUniformItemSizes(True)
        self.list.setWordWrap(True)
        self.list.setSpacing(4)
        self.list.setMinimumHeight(190)
        blank = QPixmap(self.THUMB, self.THUMB)
        blank.fill(QColor("#2d2d33"))
        self._items: dict[str, QListWidgetItem] = {}
        for key in frame_keys():
            item = QListWidgetItem(QIcon(blank), FRAMES[key].label)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setToolTip(f"{FRAMES[key].group}: {FRAMES[key].descricao}")
            self.list.addItem(item)
            self._items[key] = item
        lay.addWidget(self.list, 1)

        self.lbl_desc = QLabel("Escolha uma moldura.")
        self.lbl_desc.setObjectName("hint")
        self.lbl_desc.setWordWrap(True)
        lay.addWidget(self.lbl_desc)

        form = QFormLayout()
        esp_row, self.slider = _slider_row(0, 100, 50)
        self.slider.setToolTip("Espessura da borda ou intensidade do efeito. 50 = padrão da moldura.")
        form.addRow("Espessura", esp_row)
        self.btn_color = QPushButton()
        self._color = "#ffffff"
        style_color_button(self.btn_color, self._color)
        self.chk_default_color = QCheckBox("Cor própria da moldura")
        self.chk_default_color.setChecked(True)
        self.lbl_cor = QLabel("Cor")
        form.addRow(self.chk_default_color)
        form.addRow(self.lbl_cor, self.btn_color)
        self.edt_text = QLineEdit()
        self.lbl_text = QLabel("Texto")
        form.addRow(self.lbl_text, self.edt_text)
        lay.addLayout(form)

        btns = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_apply = QPushButton("Aplicar moldura")
        self.btn_apply.setObjectName("primary")
        btns.addWidget(self.btn_cancel)
        btns.addStretch()
        btns.addWidget(self.btn_apply)
        lay.addLayout(btns)

        self._timer = debounce(self, 60, self._emit_preview)
        self.list.currentItemChanged.connect(lambda *_: self._on_select())
        self.slider.valueChanged.connect(lambda _=0: self._timer.start())
        self.chk_default_color.toggled.connect(lambda _: (self._sync(), self._timer.start()))
        self.edt_text.textChanged.connect(lambda _: self._timer.start())
        self.btn_color.clicked.connect(self._pick_color)
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_cancel.clicked.connect(self._on_cancel)
        self._worker: ThumbnailWorker | None = None
        self._generation = 0
        self._sync()

    # --------------------------------------------------------------- estado
    def selected_key(self) -> str | None:
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def values(self) -> dict | None:
        key = self.selected_key()
        if not key:
            return None
        return {"nome": key, "espessura": self.slider.value(), "cor": self._color,
                "cor_padrao": self.chk_default_color.isChecked() or FRAMES[key].cor_padrao is None,
                "texto": self.edt_text.text().strip() if FRAMES[key].usa_texto else ""}

    def _sync(self) -> None:
        key = self.selected_key()
        spec = FRAMES.get(key) if key else None
        usa_cor = spec is not None and spec.cor_padrao is not None
        self.chk_default_color.setVisible(usa_cor)
        self.lbl_cor.setVisible(usa_cor)
        self.btn_color.setVisible(usa_cor)
        self.btn_color.setEnabled(usa_cor and not self.chk_default_color.isChecked())
        usa_texto = spec is not None and spec.usa_texto
        self.lbl_text.setVisible(usa_texto)
        self.edt_text.setVisible(usa_texto)
        self.edt_text.setPlaceholderText(spec.texto_dica if spec else "")
        self.btn_apply.setEnabled(spec is not None)

    def _on_select(self) -> None:
        key = self.selected_key()
        if key:
            spec = FRAMES[key]
            self.lbl_desc.setText(f"{spec.group} — {spec.descricao}")
            self.slider.setValue(50)   # cada moldura volta ao seu tamanho padrão
            self._timer.stop()         # (a prévia sai logo abaixo, sem esperar o debounce)
        self._sync()
        self._emit_preview()

    def reset(self) -> None:
        self._timer.stop()
        self.list.blockSignals(True)
        self.list.setCurrentRow(-1)
        self.list.clearSelection()
        self.list.blockSignals(False)
        self.lbl_desc.setText("Escolha uma moldura.")
        self._sync()

    def _emit_preview(self) -> None:
        v = self.values()
        if v:
            self.previewChanged.emit(v)

    def _on_apply(self) -> None:
        v = self.values()
        if v:
            self.applyRequested.emit(v)

    def _on_cancel(self) -> None:
        self.reset()
        self.cancelRequested.emit()

    def _pick_color(self) -> None:
        c = choose_color(self, self._color, "Cor da moldura")
        if c:
            self._color = c
            style_color_button(self.btn_color, c)
            self.chk_default_color.setChecked(False)
            self._timer.start()

    # ----------------------------------------------------------- miniaturas
    @staticmethod
    def render_thumb(source: np.ndarray, key: str, size: int) -> np.ndarray:
        """Moldura aplicada à miniatura, encaixada num quadrado `size` (fundo escuro)."""
        framed = apply_frame(source, key)
        h, w = framed.shape[:2]
        s = size / max(h, w)
        small = cv2.resize(framed, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        out = np.empty((size, size, 3), np.uint8)
        out[...] = (35, 35, 40)
        y0, x0 = (size - small.shape[0]) // 2, (size - small.shape[1]) // 2
        out[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        return out

    def set_source(self, img: np.ndarray | None) -> None:
        """Regera as miniaturas a partir da foto atual (em segundo plano)."""
        self._stop_worker()
        self._generation += 1
        if img is None:
            return
        dpr = max(1.0, self.devicePixelRatioF())
        size = int(self.THUMB * dpr)
        thumb, _ = downscale(img, int(300 * dpr))
        thumb = np.ascontiguousarray(thumb)
        self._worker = ThumbnailWorker(thumb, self._generation, self, keys=frame_keys(),
                                       render=lambda src, k: self.render_thumb(src, k, size))
        self._worker.thumbReady.connect(self._on_thumb)
        self._worker.start()

    def _on_thumb(self, generation: int, key: str, arr: np.ndarray) -> None:
        if generation != self._generation or key not in self._items:
            return
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
