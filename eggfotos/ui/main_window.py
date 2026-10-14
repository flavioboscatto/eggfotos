"""Janela principal do EggFotos."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import cv2
from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QDockWidget, QFileDialog, QInputDialog, QLabel, QMainWindow, QMessageBox,
    QProgressDialog, QStyle, QToolButton,
)

from .. import APP_NAME, __version__
from ..core import background, faces, geometry, inpaint, segment, watermark
from ..core.blur import blur_with_mask
from ..core.history import History
from ..errors import friendly_error
from ..core.image_io import OPEN_FILTER, is_supported, load_image, save_image
from ..pipeline.operations import apply_step, get_operation
from ..pipeline.recipe import Recipe, recipes_dir, safe_filename
from .batch_dialog import BatchDialog
from .canvas import TOOL_BRUSH, TOOL_CROP, TOOL_NAV, TOOL_POINT, TOOL_REGION, ImageCanvas
from .tool_tabs import ToolTabs
from .workers import TaskWorker
from .panels import AdjustPanel, BackgroundPanel, BlurPanel, CropPanel, FacesPanel, FiltersPanel, ResizeDialog, WatermarkPanel

PROXY_MAX = 1800  # lado maior da cópia reduzida usada nas prévias ao vivo
TAB_ADJUST, TAB_FILTERS, TAB_CROP, TAB_BLUR, TAB_BACKGROUND, TAB_FACES, TAB_WATERMARK = range(7)
SAVE_FILTER = "JPG (*.jpg);;PNG (*.png);;WEBP (*.webp);;TIFF (*.tif)"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings(APP_NAME, APP_NAME)
        self.image: np.ndarray | None = None
        self.proxy: np.ndarray | None = None
        self.proxy_scale = 1.0
        self.meta: dict = {}
        self.path: Path | None = None
        self.history = History()
        self._saved_uid: int | None = None
        self._overwrite_confirmed = False
        self._pending: Callable[[], None] | None = None  # prévia ativa (refaz ao soltar "Original")
        self._thumbs_stale = True  # miniaturas de filtro precisam ser regeradas
        self._matte: np.ndarray | None = None  # máscara do "Fundo" (resolução total)
        self._matte_key: tuple[int | None, str] | None = None  # (uid do histórico, modelo)
        self._proxy_matte: np.ndarray | None = None
        self._face_boxes: list | None = None  # rostos detectados (relativos) da foto atual
        self._face_uid: int | None = None
        self._wm_mask: np.ndarray | None = None   # marcação da aba Marca d'água (resolução total)
        self._wm_rects: list[tuple[float, float, float, float]] = []  # retângulos relativos
        self._matte_manual = False                # máscara veio de cliques (não reproduzível em lote)
        self._sam_emb: segment.Embedding | None = None  # embedding do MobileSAM da foto atual
        self._sam_uid: int | None = None
        self._sam_points: list[tuple[float, float, int]] = []  # cliques do modo Fundo (relativos)
        self._sam_pixels: list[tuple[float, float, bool]] = []  # os mesmos, em pixels (marcadores)
        self._sam_mask: np.ndarray | None = None
        self._click_busy = False
        self._wm_strokes = 0
        self._mark_undo: list[tuple] = []         # desfazer das MARCAÇÕES (traço/retângulo/clique/ajuste)
        self._mark_redo: list[tuple] = []
        self._mark_applied: dict[int, tuple] = {}  # uid do estado criado por "Remover" → marcação de antes
        self._bg_redo_points: list[tuple[float, float, int]] = []  # cliques desfeitos (modo Fundo)
        self._wm_loose: np.ndarray | None = None  # marcação solta, guardada antes do "apertar aos traços"
        self._wm_template: str | None = None      # nome do modelo salvo para a marcação atual
        self._wm_brush = False                    # houve traço de pincel (marcação não reproduzível)
        self._task: TaskWorker | None = None
        self._zombies: list[TaskWorker] = []  # threads vivas (inclui as canceladas)
        self._task_dialog: QProgressDialog | None = None

        self.canvas = ImageCanvas()
        self.setCentralWidget(self.canvas)
        self._build_panels()
        self._build_actions()
        self._build_status()
        self._connect()

        self.setAcceptDrops(True)
        self.resize(1400, 900)
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        state = self.settings.value("state")  # posição e largura da barra de ferramentas
        if state is not None:
            self.restoreState(state)
        self._update_state()

    # ================================================================ montagem
    def _build_panels(self) -> None:
        self.adjust_panel = AdjustPanel()
        self.filters_panel = FiltersPanel()
        self.crop_panel = CropPanel()
        self.blur_panel = BlurPanel()
        self.background_panel = BackgroundPanel()
        self.faces_panel = FacesPanel()
        self.wm_panel = WatermarkPanel()
        self.tabs = ToolTabs()
        self.tabs.addTab(self.adjust_panel, "Ajustes")
        self.tabs.addTab(self.filters_panel, "Filtros", scroll=False)  # a grade de filtros já rola
        self.tabs.addTab(self.crop_panel, "Recortar")
        self.tabs.addTab(self.blur_panel, "Borrar")
        self.tabs.addTab(self.background_panel, "Fundo")
        self.tabs.addTab(self.faces_panel, "Ocultar rosto")
        self.tabs.addTab(self.wm_panel, "Marca d'água")
        dock = QDockWidget("Ferramentas", self)
        dock.setObjectName("dock_tools")
        dock.setWidget(self.tabs)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        dock.setMinimumWidth(330)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.resizeDocks([dock], [400], Qt.Orientation.Horizontal)  # largura inicial; depois o usuário ajusta
        self.dock_tools = dock

    def _action(self, text: str, slot, shortcut=None, icon=None, tip: str = "") -> QAction:
        a = QAction(text, self)
        if shortcut is not None:
            if isinstance(shortcut, (list, tuple)):
                a.setShortcuts([QKeySequence(s) for s in shortcut])
            else:
                a.setShortcut(QKeySequence(shortcut))
        if icon is not None:
            a.setIcon(self.style().standardIcon(icon))
        if tip:
            a.setToolTip(tip)
        a.triggered.connect(slot)
        return a

    def _build_actions(self) -> None:
        SP = QStyle.StandardPixmap
        self.act_open = self._action("Abrir…", self.open_dialog, QKeySequence.StandardKey.Open, SP.SP_DialogOpenButton)
        self.act_save = self._action("Salvar", self.save, QKeySequence.StandardKey.Save, SP.SP_DialogSaveButton)
        self.act_save_as = self._action("Salvar como…", self.save_as, "Ctrl+Shift+S")
        self.act_batch = self._action("Processar pasta…", self.open_batch, "Ctrl+B", SP.SP_DirIcon,
                                      "Aplicar uma receita a todas as fotos de uma pasta")
        self.act_quit = self._action("Sair", self.close, QKeySequence.StandardKey.Quit)
        self.act_undo = self._action("Desfazer", self.undo, QKeySequence.StandardKey.Undo, SP.SP_ArrowBack)
        self.act_redo = self._action("Refazer", self.redo, ["Ctrl+Y", "Ctrl+Shift+Z"], SP.SP_ArrowForward)
        self.act_recipe = self._action("Salvar edições como receita…", self.save_edits_as_recipe)
        self.act_rot_r = self._action("Girar 90° à direita", lambda: self.rotate(90), "Ctrl+R")
        self.act_rot_l = self._action("Girar 90° à esquerda", lambda: self.rotate(270), "Ctrl+Shift+R")
        self.act_flip_h = self._action("Espelhar horizontal", lambda: self.flip("horizontal"))
        self.act_flip_v = self._action("Espelhar vertical", lambda: self.flip("vertical"))
        self.act_resize = self._action("Redimensionar…", self.resize_dialog, "Ctrl+Alt+I")
        self.act_fit = self._action("Ajustar à janela", self.canvas.fit_in_view, "Ctrl+0")
        self.act_100 = self._action("Tamanho real (100%)", self.canvas.zoom_100, "Ctrl+1")
        self.act_zoom_in = self._action("Aumentar zoom", lambda: self.canvas.zoom_by(1.25),
                                       [QKeySequence.StandardKey.ZoomIn, "Ctrl+="])
        self.act_zoom_out = self._action("Diminuir zoom", lambda: self.canvas.zoom_by(0.8), QKeySequence.StandardKey.ZoomOut)
        self.act_about = self._action("Sobre o EggFotos", self.about)

        mb = self.menuBar()
        m = mb.addMenu("&Arquivo")
        for a in (self.act_open, self.act_save, self.act_save_as, None, self.act_batch, None, self.act_quit):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("&Editar")
        for a in (self.act_undo, self.act_redo, None, self.act_recipe):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("&Imagem")
        for a in (self.act_rot_r, self.act_rot_l, self.act_flip_h, self.act_flip_v, None, self.act_resize):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("E&xibir")
        for a in (self.act_fit, self.act_100, self.act_zoom_in, self.act_zoom_out):
            m.addAction(a)
        mb.addMenu("Aj&uda").addAction(self.act_about)

        tb = self.addToolBar("Principal")
        tb.setObjectName("toolbar_main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        for a in (self.act_open, self.act_save, None, self.act_undo, self.act_redo, None,
                  self.act_fit, self.act_100):
            tb.addSeparator() if a is None else tb.addAction(a)
        tb.addSeparator()
        self.btn_compare = QToolButton()
        self.btn_compare.setText("◐ Original")
        self.btn_compare.setCheckable(True)
        self.btn_compare.setToolTip("Clique para ver a foto original; clique de novo para voltar à edição")
        tb.addWidget(self.btn_compare)
        tb.addSeparator()
        tb.addAction(self.act_batch)

    def _build_status(self) -> None:
        self.lbl_info = QLabel("Abra uma foto (Ctrl+O) ou arraste para a janela")
        self.lbl_zoom = QLabel("")
        self.statusBar().addWidget(self.lbl_info, 1)
        for act, text, tip in ((self.act_zoom_out, "−", "Diminuir zoom (tecla −)"),
                               (self.act_zoom_in, "+", "Aumentar zoom (tecla +)"),
                               (self.act_fit, "Ajustar", "Ajustar à janela (Ctrl+0)"),
                               (self.act_100, "100%", "Tamanho real (Ctrl+1)")):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.clicked.connect(act.trigger)
            act.changed.connect(lambda a=act, btn=b: btn.setEnabled(a.isEnabled()))
            b.setEnabled(act.isEnabled())
            self.statusBar().addPermanentWidget(b)
        self.statusBar().addPermanentWidget(self.lbl_zoom)

    def _connect(self) -> None:
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.canvas.zoomChanged.connect(lambda z: self.lbl_zoom.setText(f"{z * 100:.0f}%"))

        self.adjust_panel.previewChanged.connect(self._preview_adjust)
        self.adjust_panel.applyRequested.connect(self._apply_adjust)
        self.adjust_panel.cancelRequested.connect(self._cancel_preview)

        self.filters_panel.previewChanged.connect(self._preview_filter)
        self.filters_panel.applyRequested.connect(self._apply_filter)
        self.filters_panel.cancelRequested.connect(self._cancel_preview)

        self.crop_panel.ratioChanged.connect(self._on_ratio)
        self.crop_panel.applyCrop.connect(self._apply_crop)
        self.crop_panel.clearCrop.connect(self.canvas.clear_crop)
        self.crop_panel.rotate90.connect(self.rotate)
        self.crop_panel.flip.connect(self.flip)
        self.crop_panel.straightenPreview.connect(self._preview_straighten)
        self.crop_panel.straightenApply.connect(self._apply_straighten)
        self.canvas.cropChanged.connect(self.crop_panel.set_selection)
        self.canvas.cropAccepted.connect(self._apply_crop)

        self.blur_panel.settingsChanged.connect(self._sync_blur_tool)
        self.canvas.brushStrokeFinished.connect(self._apply_brush_blur)
        self.canvas.regionSelected.connect(self._apply_region_blur)

        bp = self.background_panel
        bp.downloadRequested.connect(self._download_model)
        bp.detectRequested.connect(self._detect_background)
        bp.previewChanged.connect(self._preview_background)
        bp.applyRequested.connect(self._apply_background)
        bp.cancelRequested.connect(self._cancel_preview)
        bp.clickModeRequested.connect(self._bg_click_start)
        bp.clickUse.connect(self._bg_click_use)
        bp.clickClear.connect(self._bg_click_clear)
        bp.clickExit.connect(self._sam_reset)
        self.canvas.pointClicked.connect(self._on_point_clicked)

        fp = self.faces_panel
        fp.downloadRequested.connect(self._download_face_model)
        fp.detectRequested.connect(self._detect_faces)
        fp.previewChanged.connect(self._preview_faces)
        fp.applyRequested.connect(self._apply_faces)
        fp.cancelRequested.connect(self._cancel_preview)

        wp = self.wm_panel
        wp.settingsChanged.connect(self._sync_wm_tool)
        wp.clearRequested.connect(self._wm_clear)
        wp.applyRequested.connect(self._apply_watermark)
        wp.downloadRequested.connect(self._download_lama)
        wp.snapRequested.connect(self._wm_snap)
        wp.snapUndoRequested.connect(self._wm_snap_undo)
        wp.saveTemplateRequested.connect(self._wm_save_template)
        wp.learnRequested.connect(self._wm_learn)

        self.btn_compare.toggled.connect(lambda on: self._show_original() if on else self._hide_original())

    # ============================================================= estado
    def _is_modified(self) -> bool:
        return self.image is not None and self.history.current_uid != self._saved_uid

    def _update_state(self) -> None:
        has = self.image is not None
        for a in (self.act_save, self.act_save_as, self.act_rot_r, self.act_rot_l, self.act_flip_h,
                  self.act_flip_v, self.act_resize, self.act_fit, self.act_100, self.act_zoom_in,
                  self.act_zoom_out, self.act_recipe):
            a.setEnabled(has)
        self.tabs.setEnabled(has)
        self.btn_compare.setEnabled(has and self.history.can_undo())
        self._refresh_undo_actions()
        name = self.path.name if self.path else ""
        mod = " •" if self._is_modified() else ""
        self.setWindowTitle(f"{name}{mod} — {APP_NAME}" if name else APP_NAME)
        if has:
            h, w = self.image.shape[:2]
            alpha = " · transparência" if self.image.shape[2] == 4 else ""
            self.lbl_info.setText(f"{name}   {w} × {h} px{alpha}")

    def _set_working_image(self, img: np.ndarray, reset_view: bool = False) -> None:
        """Troca a imagem de trabalho e atualiza tudo que depende dela."""
        self.image = img
        self.proxy, self.proxy_scale = geometry.downscale(img, PROXY_MAX)
        self._pending = None
        self._uncheck_original()
        self.canvas.set_image(img, reset_view)
        self.adjust_panel.reset()
        self.filters_panel.reset()
        self.crop_panel.reset_angle()
        self._drop_matte()
        self._face_boxes = self._face_uid = None
        self.faces_panel.reset()
        self._wm_reset()
        self._sam_reset()
        if self.tabs.currentIndex() == TAB_FILTERS:
            self.filters_panel.set_source(img)
        else:
            self._thumbs_stale = True
        self._update_state()

    def commit(self, new_img: np.ndarray, label: str, step: dict | None = None) -> None:
        """Grava uma edição no histórico. `step` = receita equivalente (ou None se manual)."""
        self.history.push(new_img, label, step)
        self._set_working_image(new_img)
        self.statusBar().showMessage(f"✓ {label}", 2500)

    def _busy(self, func: Callable[[], np.ndarray]) -> np.ndarray | None:
        """Roda na própria thread da UI, com cursor de espera. Só para o que precisa do resultado
        na hora (salvar arquivo, que _confirm_discard e o fechamento esperam); o resto usa _run_task."""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            return func()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, f"Não foi possível concluir a operação:\n{friendly_error(e)}")
            return None
        finally:
            QApplication.restoreOverrideCursor()

    def _apply_op(self, step: dict, label: str | None = None) -> None:
        """Aplica um step reproduzível à foto inteira e grava no histórico."""
        if self.image is None:
            return
        self._edit_async(lambda img: apply_step(img, step),
                         label or get_operation(step["op"]).label, step)

    # ============================================================= prévias
    def _run_preview(self, render: Callable[[], None]) -> None:
        if self.image is None or self._working:
            return
        self._uncheck_original()
        self._pending = render
        try:
            render()
        except Exception as e:  # noqa: BLE001
            self.statusBar().showMessage(f"Erro na prévia: {friendly_error(e)}", 4000)

    def _cancel_preview(self) -> None:
        if self._pending is not None and self.image is not None:
            self._pending = None
            self.canvas.set_image(self.image)

    def _preview_step(self, step: dict, centered: bool = False) -> None:
        def render():
            out = apply_step(self.proxy, step)
            self.canvas.set_preview(out, 1.0 / self.proxy_scale, centered)
        self._run_preview(render)

    def _uncheck_original(self) -> None:
        """Sai do modo "ver original" sem redesenhar (quem chama já cuida da tela)."""
        if self.btn_compare.isChecked():
            self.btn_compare.blockSignals(True)
            self.btn_compare.setChecked(False)
            self.btn_compare.blockSignals(False)

    def _show_original(self) -> None:
        if self.history.original is not None:
            self.canvas.set_image(self.history.original)
            self.statusBar().showMessage("Mostrando a foto original — clique em Original para voltar.")

    def _hide_original(self) -> None:
        if self.image is None:
            return
        self.canvas.set_image(self.image)
        if self._pending is not None:
            self._run_preview(self._pending)

    # ============================================================== abas
    def _on_tab_changed(self, idx: int) -> None:
        if self.btn_compare.isChecked() and self.image is not None:  # sai do "ver original"
            self._uncheck_original()
            self.canvas.set_image(self.image)
        self._cancel_preview()
        self._sam_reset()
        self._refresh_undo_actions()
        self.adjust_panel.reset()
        self.filters_panel.reset()
        self.crop_panel.reset_angle()
        if idx == TAB_BACKGROUND:
            self.background_panel.refresh_model_status()
        if idx == TAB_FACES:
            self.faces_panel.refresh_model_status()
        if idx == TAB_WATERMARK:
            self.wm_panel.refresh_model_status()
        if idx == TAB_FILTERS and self._thumbs_stale and self.image is not None:
            self.filters_panel.set_source(self.image)
            self._thumbs_stale = False
        if idx == TAB_CROP:
            self.canvas.set_tool(TOOL_CROP)
            self._on_ratio(self.crop_panel.ratio_key())
        elif idx == TAB_BLUR:
            self._sync_blur_tool()
        elif idx == TAB_WATERMARK:
            self._sync_wm_tool()
        else:
            self.canvas.set_tool(TOOL_NAV)

    # ============================================================= ajustes
    def _preview_adjust(self, values: dict) -> None:
        self._preview_step({"op": "ajustes", **values})

    def _apply_adjust(self, values: dict) -> None:
        if any(values.values()):
            self._apply_op({"op": "ajustes", **values}, "Ajustes")

    # ============================================================= filtros
    def _preview_filter(self, key: str, intensity: int) -> None:
        self._preview_step({"op": "filtro", "nome": key, "intensidade": intensity})

    def _apply_filter(self, key: str, intensity: int) -> None:
        from ..core.filters import filter_label
        self._apply_op({"op": "filtro", "nome": key, "intensidade": intensity},
                       f"Filtro {filter_label(key)}")

    # ============================================================= recorte
    def _on_ratio(self, key: str) -> None:
        if key == "original" and self.image is not None:
            h, w = self.image.shape[:2]
            self.canvas.set_crop_ratio(w / h)
        else:
            self.canvas.set_crop_ratio(geometry.parse_ratio(key))

    def _apply_crop(self) -> None:
        rect = self.canvas.crop_rect()
        if rect is None or self.image is None:
            self.statusBar().showMessage("Arraste sobre a foto para selecionar a área.", 3000)
            return
        H, W = self.image.shape[:2]
        x, y, w, h = rect
        step = {"op": "recortar_relativo", "x": x / W, "y": y / H, "w": w / W, "h": h / H}
        if self._guard():
            return
        out = geometry.crop(self.image, x, y, w, h)  # pixels exatos; é só fatiar, instantâneo
        self.canvas.clear_crop()
        self.commit(out, "Recortar", step)

    def rotate(self, angle: int) -> None:
        self._apply_op({"op": "girar", "angulo": angle}, f"Girar {angle}°")

    def flip(self, direction: str) -> None:
        self._apply_op({"op": "espelhar", "direcao": direction}, f"Espelhar {direction}")

    def _preview_straighten(self, angle: float) -> None:
        if abs(angle) < 1e-6:
            self._cancel_preview()
        else:
            self._preview_step({"op": "endireitar", "angulo": angle}, centered=True)

    def _apply_straighten(self, angle: float) -> None:
        if abs(angle) > 1e-6:
            self._apply_op({"op": "endireitar", "angulo": angle}, f"Endireitar {angle:+.1f}°")

    def resize_dialog(self) -> None:
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        dlg = ResizeDialog(w, h, self)
        if not dlg.exec():
            return
        nw, nh = dlg.size_values()
        if (nw, nh) == (w, h):
            return
        proportional = abs(nw / nh - w / h) < 0.01
        if proportional:  # proporcional = reproduzível no lote
            self._apply_op({"op": "redimensionar", "lado_maior": max(nw, nh), "ampliar": True},
                           f"Redimensionar {nw}×{nh}")
        else:
            self._edit_async(lambda img: geometry.resize(img, nw, nh), f"Redimensionar {nw}×{nh}", None)

    # ============================================================== borrar
    def _sync_blur_tool(self) -> None:
        if self.tabs.currentIndex() != TAB_BLUR:
            return
        tool = self.blur_panel.tool()
        if tool == "pincel":
            self.canvas.set_tool(TOOL_BRUSH)
            self.canvas.set_brush_size(self.blur_panel.brush_size())
        else:
            self.canvas.region_shape = tool
            self.canvas.set_tool(TOOL_REGION)

    def _apply_brush_blur(self, mask: np.ndarray) -> None:
        if self.image is None:
            return
        if self.tabs.currentIndex() == TAB_WATERMARK:
            self._wm_add_mask(mask, brush=True)
            return
        bp = self.blur_panel
        tipo, inten, cor = bp.tipo(), bp.intensidade(), bp.cor()
        self._edit_async(lambda img: blur_with_mask(img, mask, tipo, inten, cor), "Borrar (pincel)", None)

    def _apply_region_blur(self, rect) -> None:
        if self.image is None:
            return
        if self.tabs.currentIndex() == TAB_WATERMARK:
            H, W = self.image.shape[:2]
            x, y, w, h = rect
            self._wm_add_mask(inpaint.rect_mask(self.image.shape, x / W, y / H, w / W, h / H),
                              brush=False, rect=(x / W, y / H, w / W, h / H))
            return
        H, W = self.image.shape[:2]
        x, y, w, h = rect
        bp = self.blur_panel
        step = {"op": "borrar_regiao", "x": x / W, "y": y / H, "w": w / W, "h": h / H,
                "forma": self.canvas.region_shape, "tipo": bp.tipo(),
                "intensidade": bp.intensidade(), "cor": bp.cor()}
        self._apply_op(step, "Borrar região")

    # ============================================================== fundo
    def _drop_matte(self) -> None:
        self._matte = self._proxy_matte = self._matte_key = None
        self._matte_manual = False
        self.background_panel.reset()

    @property
    def _working(self) -> bool:
        return self._task is not None

    def _guard(self) -> bool:
        """True (e avisa) se há uma operação em andamento: ações que mexem na foto esperam."""
        if self._task is not None:
            self.statusBar().showMessage("Aguarde: operação em andamento…", 2500)
            return True
        return False

    def _run_task(self, text: str, func: Callable[[], object], on_done: Callable[[object], None],
                  cancelable: bool = True) -> bool:
        """Roda `func` numa QThread. Cursor de espera na hora; a janelinha com Cancelar aparece se
        passar de 0,3 s. Cancelar descarta o resultado (a conta termina em segundo plano).
        Devolve False se já havia outra operação rodando."""
        if self._guard():
            return False
        dlg = QProgressDialog(text, "Cancelar" if cancelable else "", 0, 0, self)
        if not cancelable:
            dlg.setCancelButton(None)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setWindowTitle(APP_NAME)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        task = TaskWorker(func, self)
        state = {"cancelled": False, "closed": False}

        def finish():
            if state["closed"]:
                return
            state["closed"] = True
            QApplication.restoreOverrideCursor()
            dlg.close()
            dlg.deleteLater()
            if self._task is task:
                self._task = self._task_dialog = None

        def ok(result):
            finish()
            if not state["cancelled"]:
                on_done(result)

        def fail(msg):
            finish()
            if not state["cancelled"]:
                QMessageBox.warning(self, APP_NAME, msg)

        def cancel():
            if state["closed"]:  # o QProgressDialog também emite "canceled" ao ser fechado por nós
                return
            state["cancelled"] = True
            finish()
            self.statusBar().showMessage("Cancelado.", 2500)

        def release():  # a thread terminou de verdade
            if task in self._zombies:
                self._zombies.remove(task)
            task.deleteLater()

        self._zombies.append(task)  # referência até a thread acabar (evita destruir QThread rodando)
        task.finished.connect(release)
        task.done.connect(ok)
        task.failed.connect(fail)
        dlg.canceled.connect(cancel)
        self._task, self._task_dialog = task, dlg
        task.start()
        QTimer.singleShot(300, lambda: (not state["closed"]) and dlg.show())
        return True

    def _edit_async(self, func: Callable[[np.ndarray], np.ndarray], label: str, step: dict | None) -> None:
        """Aplica `func(foto)` em segundo plano e grava o resultado no histórico (se a foto não mudou)."""
        if self.image is None:
            return
        img, uid = self.image, self.history.current_uid

        def done(out):
            if self.history.current_uid == uid:
                self.commit(out, label, step)

        self._run_task(f"{label}…", lambda: func(img), done)

    def _download_model(self, modelo: str) -> None:
        nome, rotulo = background.MODELS[modelo]
        r = QMessageBox.question(
            self, APP_NAME,
            f"Baixar o modelo '{rotulo}' ({nome}.onnx)?\n\n"
            "Origem: github.com/danielgatis/rembg (releases)\n"
            f"Destino: {background.models_dir()}\n\n"
            "É feito uma única vez.",
        )
        if r != QMessageBox.StandardButton.Yes:
            return

        def done(_):
            self.background_panel.refresh_model_status()
            self.statusBar().showMessage("✓ Modelo baixado", 3000)

        self._run_task("Baixando o modelo… pode levar alguns minutos.",
                       lambda: background.download_model(modelo), done)

    def _detect_background(self, modelo: str) -> None:
        if self.image is None:
            return
        img, key = self.image, (self.history.current_uid, modelo)

        def done(matte):
            if self.history.current_uid != key[0]:  # a foto mudou enquanto detectava
                return
            self._matte, self._matte_key = matte, key
            self._matte_manual = False
            self._proxy_matte = None
            self.background_panel.set_detected(True)
            self._preview_background(self.background_panel.values())

        self._run_task("Detectando o objeto…", lambda: background.compute_matte(img, modelo), done)

    def _matte_ready(self, modelo: str) -> bool:
        return (self._matte is not None and self._matte_key is not None
                and self._matte_key[0] == self.history.current_uid
                and (self._matte_manual or self._matte_key[1] == modelo))

    def _background_step(self, values: dict) -> dict:
        """Step reproduzível correspondente ao estado do painel Fundo."""
        if values["fundo"] == "desfoque":
            return get_operation("desfocar_fundo").new_step(
                modelo=values["modelo"], intensidade=values["desfoque"],
                encolher=values["encolher"], suavizar=values["suavizar"])
        step = get_operation("remover_fundo").new_step(
            **{k: v for k, v in values.items() if k != "desfoque"})
        return step

    def _render_background(self, img: np.ndarray, matte: np.ndarray, values: dict) -> np.ndarray:
        if values["fundo"] == "desfoque":
            return background.blur_background(
                img, matte, values["desfoque"], values["encolher"], values["suavizar"])
        params = {k: v for k, v in values.items() if k not in ("modelo", "desfoque")}
        return background.apply_matte(img, matte, **params)

    def _preview_background(self, values: dict) -> None:
        if self.image is None or not self._matte_ready(values["modelo"]):
            return
        if self._proxy_matte is None or self._proxy_matte.shape[:2] != self.proxy.shape[:2]:
            ph, pw = self.proxy.shape[:2]
            self._proxy_matte = cv2.resize(self._matte, (pw, ph), interpolation=cv2.INTER_AREA)

        def render():
            out = self._render_background(self.proxy, self._proxy_matte, values)
            self.canvas.set_preview(out, 1.0 / self.proxy_scale)
        self._run_preview(render)

    def _apply_background(self, values: dict) -> None:
        if self.image is None or not self._matte_ready(values["modelo"]):
            return
        step = None if self._matte_manual else self._background_step(values)  # cliques: só nesta foto
        label = "Desfocar fundo" if values["fundo"] == "desfoque" else "Remover fundo"
        img, matte = self.image, self._matte
        self._run_task("Aplicando…", lambda: self._render_background(img, matte, values),
                       lambda out: self.commit(out, label, step))

    # ============================================================= rostos
    def _download_face_model(self) -> None:
        r = QMessageBox.question(
            self, APP_NAME,
            f"Baixar o modelo de rostos YuNet ({faces.MODEL_FILE}, ~230 KB)?\n\n"
            "Origem: github.com/opencv/opencv_zoo\n"
            f"Destino: {background.models_dir()}",
        )
        if r != QMessageBox.StandardButton.Yes:
            return

        def done(_):
            self.faces_panel.refresh_model_status()
            self.statusBar().showMessage("✓ Modelo de rostos baixado", 3000)

        self._run_task("Baixando o modelo de rostos…", faces.download_model, done)

    def _detect_faces(self) -> None:
        if self.image is None:
            return
        img, uid = self.image, self.history.current_uid

        def done(boxes):
            if self.history.current_uid != uid:
                return
            self._face_boxes, self._face_uid = boxes, uid
            n = len(boxes)
            self.faces_panel.set_detected(n > 0, n)
            if n:
                self._preview_faces(self.faces_panel.values())

        self._run_task("Procurando rostos…", lambda: faces.detect_faces(img), done)

    def _preview_faces(self, values: dict) -> None:
        if self.image is None or not self._face_boxes or self._face_uid != self.history.current_uid:
            return

        def render():
            out = faces.blur_boxes(self.proxy, self._face_boxes, **values)
            self.canvas.set_preview(out, 1.0 / self.proxy_scale)
        self._run_preview(render)

    def _apply_faces(self, values: dict) -> None:
        if self.image is None or not self._face_boxes or self._face_uid != self.history.current_uid:
            return
        step = get_operation("desfocar_rostos").new_step(**values)
        img, boxes = self.image, self._face_boxes
        self._run_task("Aplicando…", lambda: faces.blur_boxes(img, boxes, **values),
                       lambda out: self.commit(out, "Ocultar rostos", step))

    # ================================================ seleção por cliques (SAM)
    def _show_mask_overlay(self, mask: np.ndarray, color: tuple[int, int, int]) -> None:
        """Pinta a máscara em cor translúcida sobre a prévia da foto."""
        def render():
            ph, pw = self.proxy.shape[:2]
            m = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST) > 0
            out = self.proxy.copy()
            if out.shape[2] == 3:
                out[m] = (out[m].astype(np.float32) * 0.4 + np.array(color, np.float32) * 0.6).astype(np.uint8)
            self.canvas.set_preview(out, 1.0 / self.proxy_scale)
        self._run_preview(render)

    def _sam_reset(self) -> None:
        """Sai do modo de cliques e limpa marcadores/seleção temporária."""
        self._sam_points, self._sam_pixels, self._sam_mask = [], [], None
        self._bg_redo_points = []
        self.canvas.clear_point_markers()
        self.background_panel.set_click_mode(False)
        if self.tabs.currentIndex() == TAB_BACKGROUND and self.canvas.tool() == TOOL_POINT:
            self.canvas.set_tool(TOOL_NAV)
            self._cancel_preview()

    def _ensure_sam_models(self, then: Callable[[], None]) -> None:
        """Garante os modelos (pede confirmação para baixar) e segue com `then`."""
        if segment.model_available():
            then()
            return
        r = QMessageBox.question(
            self, APP_NAME,
            "Baixar os modelos de seleção por clique (MobileSAM, ~45 MB)?\n\n"
            "Origem: huggingface.co/Acly/MobileSAM\n"
            f"Destino: {background.models_dir()}\n\nÉ feito uma única vez.",
        )
        if r == QMessageBox.StandardButton.Yes:
            self._run_task("Baixando os modelos de seleção…", segment.download_models,
                           lambda _: then())

    def _sam_segment(self, points: list, then: Callable[[np.ndarray], None]) -> None:
        """Calcula a máscara dos pontos (embedding só na 1ª vez por foto) numa thread."""
        if self.image is None or self._click_busy:
            return
        img, uid = self.image, self.history.current_uid
        cached = self._sam_emb if self._sam_uid == uid else None

        def job():
            emb = cached if cached is not None else segment.embed(img)
            return emb, segment.predict(emb, points)

        def done(result):
            emb, mask = result
            if self.history.current_uid != uid:
                return
            self._sam_emb, self._sam_uid = emb, uid
            then(mask)

        self._run_task("Analisando a foto…" if cached is None else "Selecionando…", job, done)

    def _on_point_clicked(self, x: float, y: float, positive: bool) -> None:
        if self.image is None:
            return
        H, W = self.image.shape[:2]
        tab = self.tabs.currentIndex()
        if tab == TAB_BACKGROUND:
            self._sam_points.append((x / W, y / H, 1 if positive else 0))
            self._sam_pixels.append((x, y, positive))
            self._bg_redo_points = []
            self._refresh_undo_actions()
            self._ensure_sam_models(self._bg_click_run)
        elif tab == TAB_WATERMARK:
            def go():
                self._sam_segment([(x / W, y / H, 1)], lambda m: self._wm_click_done(m, positive))
            self._ensure_sam_models(go)

    # --- aba Fundo
    def _bg_click_start(self) -> None:
        if self.image is None:
            return
        self._sam_points, self._sam_pixels, self._sam_mask = [], [], None
        self.background_panel.set_click_mode(True)
        self.canvas.set_tool(TOOL_POINT)

    def _bg_click_run(self) -> None:
        if not self._sam_points:
            return
        pts = list(self._sam_points)

        def done(mask):
            if pts != self._sam_points:  # os cliques mudaram enquanto calculava: refaz com os atuais
                self._bg_click_refresh()
                return
            self._sam_mask = mask
            self.canvas.set_point_markers(self._sam_pixels)
            self._show_mask_overlay(mask, (60, 140, 255))

        self._sam_segment(pts, done)

    def _bg_click_refresh(self) -> None:
        """Depois de desfazer/refazer um clique: recalcula a seleção com os cliques restantes."""
        self.canvas.set_point_markers(self._sam_pixels)
        if self._sam_points:
            self._bg_click_run()
        else:
            self._sam_mask = None
            self._cancel_preview()

    def _bg_click_clear(self) -> None:
        self._bg_redo_points = []
        self._sam_points, self._sam_pixels, self._sam_mask = [], [], None
        self.canvas.clear_point_markers()
        self._cancel_preview()

    def _bg_click_use(self) -> None:
        if self._sam_mask is None or self.image is None:
            self.statusBar().showMessage("Clique no objeto primeiro.", 3000)
            return
        self._matte, self._proxy_matte = self._sam_mask, None
        self._matte_key = (self.history.current_uid, self.background_panel.modelo())
        self._matte_manual = True
        self.background_panel.set_detected(True)
        self._sam_reset()
        self._preview_background(self.background_panel.values())

    # --- aba Marca d'água
    def _wm_click_done(self, mask: np.ndarray, positive: bool) -> None:
        if positive:
            self._wm_add_mask(mask, brush=True)
        elif self._wm_mask is not None:  # Alt+clique: tira o objeto da marcação
            self._wm_checkpoint()
            self._wm_mask = np.where(mask > 0, 0, self._wm_mask).astype(np.uint8)
            self._wm_show()

    # ======================================================== marca d'água
    def _wm_reset(self) -> None:
        self._mark_undo, self._mark_redo = [], []
        self._wm_mask = None
        self._wm_rects = []
        self._wm_brush = False
        self._wm_strokes = 0
        self._wm_template = None
        self._wm_loose = None
        self.wm_panel.set_snap_undo(False)
        self.wm_panel.set_template_name(None)
        self.wm_panel.set_marked(0)

    def _wm_count(self) -> int:
        return len(self._wm_rects) + self._wm_strokes

    def _sync_wm_tool(self) -> None:
        if self.tabs.currentIndex() != TAB_WATERMARK:
            return
        tool = self.wm_panel.tool()
        if tool == "pincel":
            self.canvas.set_tool(TOOL_BRUSH)
            self.canvas.set_brush_size(self.wm_panel.brush_size())
        elif tool == "clique":
            self.canvas.set_tool(TOOL_POINT)
        else:
            self.canvas.region_shape = "retangulo"
            self.canvas.set_tool(TOOL_REGION)

    # --- desfazer/refazer das marcações (passo a passo) -------------------------
    def _wm_snapshot(self) -> tuple:
        def pack(m):
            return None if m is None else (np.packbits(m > 0), m.shape)   # 1 bit/pixel: barato
        return (pack(self._wm_mask), list(self._wm_rects), self._wm_strokes, self._wm_brush,
                pack(self._wm_loose))

    def _wm_restore(self, snap: tuple) -> None:
        def unpack(p):
            if p is None:
                return None
            bits, shape = p
            n = shape[0] * shape[1]
            return (np.unpackbits(bits)[:n].reshape(shape) * 255).astype(np.uint8)
        mask, rects, strokes, brush, loose = snap
        self._wm_mask, self._wm_loose = unpack(mask), unpack(loose)
        self._wm_rects, self._wm_strokes, self._wm_brush = list(rects), strokes, brush
        self._wm_template = None
        self.wm_panel.set_template_name(None)
        self.wm_panel.set_snap_undo(self._wm_loose is not None)
        self.wm_panel.set_marked(self._wm_count())
        if self._wm_mask is None or not self._wm_mask.any():
            self._cancel_preview()
        else:
            self._wm_show()

    def _wm_checkpoint(self) -> None:
        """Guarda o estado ANTES de mudar a marcação (chame antes de cada alteração)."""
        self._mark_undo.append(self._wm_snapshot())
        del self._mark_undo[:-60]
        self._mark_redo.clear()
        self._refresh_undo_actions()

    def _marks_can_undo(self) -> bool:
        tab = self.tabs.currentIndex()
        return (tab == TAB_WATERMARK and bool(self._mark_undo)) or \
               (tab == TAB_BACKGROUND and bool(self._sam_points))

    def _marks_can_redo(self) -> bool:
        tab = self.tabs.currentIndex()
        return (tab == TAB_WATERMARK and bool(self._mark_redo)) or \
               (tab == TAB_BACKGROUND and bool(self._bg_redo_points))

    def _marks_undo(self) -> bool:
        """Desfaz UMA marcação da aba atual. True se havia o que desfazer."""
        tab = self.tabs.currentIndex()
        if tab == TAB_WATERMARK and self._mark_undo:
            self._mark_redo.append(self._wm_snapshot())
            self._wm_restore(self._mark_undo.pop())
            self.statusBar().showMessage("Marcação desfeita.", 2500)
        elif tab == TAB_BACKGROUND and self._sam_points:
            self._bg_redo_points.append(self._sam_points.pop())
            self._sam_pixels.pop()
            self._bg_click_refresh()
            self.statusBar().showMessage("Clique desfeito.", 2500)
        else:
            return False
        self._refresh_undo_actions()
        return True

    def _marks_redo(self) -> bool:
        tab = self.tabs.currentIndex()
        if tab == TAB_WATERMARK and self._mark_redo:
            self._mark_undo.append(self._wm_snapshot())
            self._wm_restore(self._mark_redo.pop())
            self.statusBar().showMessage("Marcação refeita.", 2500)
        elif tab == TAB_BACKGROUND and self._bg_redo_points:
            x, y, label = self._bg_redo_points.pop()
            H, W = self.image.shape[:2]
            self._sam_points.append((x, y, label))
            self._sam_pixels.append((x * W, y * H, label == 1))
            self._bg_click_refresh()
            self.statusBar().showMessage("Clique refeito.", 2500)
        else:
            return False
        self._refresh_undo_actions()
        return True

    def _refresh_undo_actions(self) -> None:
        """Desfazer/Refazer valem para a marcação em andamento antes de valerem para o histórico."""
        if self.image is None:
            return
        if self._marks_can_undo():
            self.act_undo.setEnabled(True)
            self.act_undo.setText("Desfazer última marcação")
        else:
            self.act_undo.setEnabled(self.history.can_undo())
            self.act_undo.setText(f"Desfazer {self.history.undo_label()}".strip())
        if self._marks_can_redo():
            self.act_redo.setEnabled(True)
            self.act_redo.setText("Refazer marcação")
        else:
            self.act_redo.setEnabled(self.history.can_redo())
            self.act_redo.setText(f"Refazer {self.history.redo_label()}".strip())

    def _wm_add_mask(self, mask: np.ndarray, brush: bool,
                     rect: tuple[float, float, float, float] | None = None) -> None:
        self._wm_checkpoint()
        if rect is not None:
            self._wm_rects.append(rect)
        self._wm_mask = mask.copy() if self._wm_mask is None else np.maximum(self._wm_mask, mask)
        if brush:
            self._wm_brush = True
            self._wm_strokes += 1
        self._wm_loose = None     # nova marcação: o "voltar ao anterior" não vale mais
        self.wm_panel.set_snap_undo(False)
        self._wm_template = None  # a marcação mudou: o modelo salvo não vale mais
        self.wm_panel.set_template_name(None)
        self.wm_panel.set_marked(self._wm_count())
        self._wm_show()

    def _wm_show(self) -> None:
        """Mostra a marcação em vermelho translúcido sobre a foto."""
        if self._wm_mask is None or self.image is None:
            return
        self._show_mask_overlay(self._wm_mask, (255, 40, 40))

    def _wm_clear(self) -> None:
        if self._wm_mask is not None:
            self._wm_checkpoint()          # "Limpar" também pode ser desfeito
            undo, redo = self._mark_undo, self._mark_redo
            self._wm_reset()
            self._mark_undo, self._mark_redo = undo, redo
        else:
            self._wm_reset()
        self._cancel_preview()
        self._refresh_undo_actions()

    def _wm_snap(self, opts: dict) -> None:
        if self._wm_mask is None or self.image is None:
            return
        img, loose = self.image, self._wm_mask

        def done(snapped):
            if not snapped.any():
                QMessageBox.information(
                    self, APP_NAME,
                    "Não encontrei traços que se destaquem do fundo nessa área. Tente aumentar a "
                    "sensibilidade, escolher a cor da marca (clara/escura) ou marcar uma área menor.")
                return
            self._wm_checkpoint()
            self._wm_loose = loose
            self._wm_mask = snapped
            self._wm_template = None
            self.wm_panel.set_template_name(None)
            self.wm_panel.set_snap_undo(True)
            self._wm_show()
            pct = 100.0 * float((snapped > 0).sum()) / max(1, int((loose > 0).sum()))
            self.statusBar().showMessage(f"Marcação apertada: agora cobre {pct:.0f}% da área marcada.", 5000)

        self._run_task("Procurando os traços da marca…",
                       lambda: inpaint.snap_to_strokes(img, loose, **opts), done)

    def _wm_snap_undo(self) -> None:
        if self._wm_loose is None:
            return
        self._wm_checkpoint()
        self._wm_mask, self._wm_loose = self._wm_loose, None
        self._wm_template = None
        self.wm_panel.set_template_name(None)
        self.wm_panel.set_snap_undo(False)
        self._wm_show()

    def _wm_save_template(self) -> None:
        if self._wm_mask is None:
            return
        nome, ok = QInputDialog.getText(self, "Salvar modelo de marca",
                                        "Nome do modelo (ex.: logo-canto):", text="logo")
        if not ok or not nome.strip():
            return
        try:
            watermark.save_mask_template(nome, watermark.make_mask_template(self._wm_mask))
        except ValueError as e:
            QMessageBox.warning(self, APP_NAME, str(e))
            return
        self._wm_template = watermark.safe_name(nome)
        self.wm_panel.set_template_name(self._wm_template)
        QMessageBox.information(
            self, APP_NAME,
            f"Modelo '{self._wm_template}' salvo em {watermark.marks_dir()}.\n\n"
            "Ao clicar em Remover, o passo entra na receita. No lote, use \"Remover marca d'água "
            "(modelo de máscara)\". A marca é ancorada ao canto mais próximo e escala com a largura.")

    def _wm_learn(self) -> None:
        pasta = QFileDialog.getExistingDirectory(
            self, "Pasta com várias fotos que têm a MESMA marca", self._last_dir())
        if not pasta:
            return
        cores = ["Branca", "Preta"]
        cor, ok = QInputDialog.getItem(self, "Cor da marca", "A marca é:", cores, 0, False)
        if not ok:
            return
        nome, ok = QInputDialog.getText(self, "Nome do modelo", "Nome do modelo:", text="marca_aprendida")
        if not ok or not nome.strip():
            return
        value = 255 if cor == "Branca" else 0

        def done(result):
            t, skipped = result
            cover = float((t.alpha > 0).mean()) * 100
            extra = f"\n{skipped} foto(s) ignorada(s) (proporção diferente ou ilegível)." if skipped else ""
            QMessageBox.information(
                self, APP_NAME,
                f"Marca aprendida com {t.n} fotos (cobre {cover:.1f}% da área).{extra}\n\n"
                f"Modelo '{watermark.safe_name(nome)}' salvo. No lote, use \"Remover marca d'água "
                "translúcida (aprendida)\". Se sobrar um resto da marca ou aparecer um contorno, ajuste a Força (80–120%).")

        self._run_task("Aprendendo a marca (pode levar alguns segundos)…",
                       lambda: watermark.learn_from_folder(pasta, nome, value), done)

    def _download_lama(self) -> None:
        r = QMessageBox.question(
            self, APP_NAME,
            f"Baixar o modelo LaMa ({inpaint.LAMA_FILE}, ~200 MB)?\n\n"
            "Origem: huggingface.co/Carve/LaMa-ONNX\n"
            f"Destino: {background.models_dir()}\n\nÉ feito uma única vez e pode levar alguns minutos.",
        )
        if r != QMessageBox.StandardButton.Yes:
            return

        def done(_):
            self.wm_panel.refresh_model_status()
            self.statusBar().showMessage("✓ Modelo LaMa baixado", 3000)

        self._run_task("Baixando o modelo LaMa… pode levar alguns minutos.", inpaint.download_lama, done)

    def _apply_watermark(self, values: dict) -> None:
        if self.image is None or self._wm_mask is None:
            return
        if values["metodo"] == "lama" and not inpaint.lama_available():
            QMessageBox.information(self, APP_NAME, "Baixe o modelo LaMa primeiro (botão na aba).")
            return
        img, mask = self.image, self._wm_mask
        # um único retângulo, sem pincel, vira receita reproduzível; o resto é edição manual
        step = None
        if self._wm_template:  # marcação salva como modelo: reproduzível em qualquer foto
            step = get_operation("remover_marca_modelo").new_step(nome=self._wm_template, **values)
        elif not self._wm_brush and len(self._wm_rects) == 1:
            x, y, w, h = self._wm_rects[0]
            step = get_operation("remover_marca_regiao").new_step(x=x, y=y, w=w, h=h, **values)
        snap = self._wm_snapshot()  # se a remoção ficar ruim, Ctrl+Z devolve as marcações para ajustar

        def done(out):
            self.commit(out, "Remover marca d'água", step)
            self._mark_applied[self.history.current_uid] = snap
            for old in list(self._mark_applied)[:-20]:
                del self._mark_applied[old]

        self._run_task("Removendo a marca d'água…",
                       lambda: inpaint.remove_with_mask(img, mask, **values), done)


    # ========================================================= histórico
    def undo(self) -> None:
        if self.tabs.currentIndex() == TAB_BACKGROUND and self._marks_undo():
            return                      # cliques da seleção: valem mesmo durante o cálculo
        if self._guard():
            return
        if self._marks_undo():          # primeiro as marcações em andamento, uma por vez
            return
        if self.history.can_undo():
            label = self.history.undo_label()
            undone_uid = self.history.current_uid
            self._set_working_image(self.history.undo())
            snap = self._mark_applied.pop(undone_uid, None)
            if snap is not None and self.tabs.currentIndex() == TAB_WATERMARK:
                self._wm_restore(snap)  # desfez uma remoção: as marcações voltam para ajustar
            self.statusBar().showMessage(f"Desfeito: {label}", 2500)
            self._refresh_undo_actions()

    def redo(self) -> None:
        if self.tabs.currentIndex() == TAB_BACKGROUND and self._marks_redo():
            return
        if self._guard():
            return
        if self._marks_redo():
            return
        if self.history.can_redo():
            label = self.history.redo_label()
            self._set_working_image(self.history.redo())
            self.statusBar().showMessage(f"Refeito: {label}", 2500)

    # ============================================================ arquivos
    def _confirm_discard(self) -> bool:
        if not self._is_modified():
            return True
        r = QMessageBox.question(
            self, APP_NAME, "Há edições não salvas. Deseja salvar antes?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if r == QMessageBox.StandardButton.Save:
            return self.save()
        return r == QMessageBox.StandardButton.Discard

    def _last_dir(self) -> str:
        return str(self.settings.value("last_dir", str(Path.home() / "Pictures")))

    def open_dialog(self) -> None:
        if self._guard() or not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Abrir foto", self._last_dir(), OPEN_FILTER)
        if path:
            self.open_file(path)

    def open_file(self, path: str | Path) -> None:
        if self._guard():
            return
        path = Path(path)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            img, meta = load_image(path)
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, APP_NAME, f"Não foi possível abrir {path.name}:\n{friendly_error(e)}")
            return
        QApplication.restoreOverrideCursor()
        self.path, self.meta = path, meta
        self.settings.setValue("last_dir", str(path.parent))
        self.history.reset(img)
        self._saved_uid = self.history.current_uid
        self._overwrite_confirmed = False
        self._thumbs_stale = True
        self._set_working_image(img, reset_view=True)
        if self.tabs.currentIndex() == TAB_FILTERS:
            self.filters_panel.set_source(img)
            self._thumbs_stale = False

    def _quality(self) -> int:
        return int(self.settings.value("quality", 92))

    def _write(self, path: Path) -> bool:
        ok = self._busy(lambda: save_image(self.image, path, quality=self._quality(), meta=self.meta))
        if ok is None:
            return False
        self.path = path
        self._saved_uid = self.history.current_uid
        self.settings.setValue("last_dir", str(path.parent))
        self.statusBar().showMessage(f"Salvo em {path}", 4000)
        self._update_state()
        return True

    def save(self) -> bool:
        if self.image is None or self._guard():
            return False
        if self.path is None:
            return self.save_as()
        if not self._overwrite_confirmed:
            r = QMessageBox.question(
                self, APP_NAME,
                f"Sobrescrever o arquivo original?\n{self.path.name}\n\n"
                "Escolha 'Não' para salvar uma cópia com outro nome.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
            )
            if r == QMessageBox.StandardButton.No:
                return self.save_as()
            if r != QMessageBox.StandardButton.Yes:
                return False
            self._overwrite_confirmed = True
        return self._write(self.path)

    def save_as(self) -> bool:
        if self.image is None or self._guard():
            return False
        base = self.path.parent if self.path else Path(self._last_dir())
        stem = f"{self.path.stem}_egg" if self.path else "foto_egg"
        ext = ".png" if self.image.shape[2] == 4 else ".jpg"
        default = str(base / f"{stem}{ext}")
        path, chosen = QFileDialog.getSaveFileName(self, "Salvar como", default, SAVE_FILTER)
        if not path:
            return False
        p = Path(path)
        if not p.suffix:  # usuário digitou o nome sem extensão: usa a do filtro escolhido
            ext = ".jpg"
            if "(*." in chosen:
                ext = "." + chosen.split("(*.", 1)[1].rstrip(")")
            p = p.with_suffix(ext)
        if p.suffix.lower() in (".jpg", ".jpeg", ".webp"):
            q, ok = QInputDialog.getInt(self, "Qualidade", "Qualidade (1–100):", self._quality(), 1, 100)
            if not ok:
                return False
            self.settings.setValue("quality", q)
            if self.image.shape[2] == 4 and p.suffix.lower() != ".webp":
                self.statusBar().showMessage("JPG não tem transparência: fundo transparente vira branco.", 5000)
        ok = self._write(p)
        if ok:
            self._overwrite_confirmed = True
        return ok

    # ============================================================== lote
    def open_batch(self) -> None:
        steps, manual = self.history.steps() if not self.history.is_empty else ([], 0)
        start = str(self.path.parent) if self.path else None
        BatchDialog(self, steps, manual, start).exec()

    def save_edits_as_recipe(self) -> None:
        steps, manual = self.history.steps()
        if not steps:
            QMessageBox.information(self, APP_NAME, "Nenhuma edição reproduzível para salvar.")
            return
        nome, ok = QInputDialog.getText(self, "Salvar receita", "Nome da receita:", text="Minha receita")
        if not ok or not nome.strip():
            return
        path = recipes_dir() / f"{safe_filename(nome.strip())}.json"
        Recipe(nome.strip(), steps).save(path)
        extra = f"\n{manual} edição(ões) manual(is) (pincel) ficaram de fora." if manual else ""
        QMessageBox.information(self, APP_NAME, f"Receita salva com {len(steps)} passo(s).{extra}")

    # ============================================================== outros
    def about(self) -> None:
        QMessageBox.about(
            self, APP_NAME,
            f"<b>{APP_NAME}</b> {__version__}<br>Editor de fotos com processamento em lote.<br><br>"
            "Uso pessoal.",
        )

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile() and is_supported(urls[0].toLocalFile()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and self._confirm_discard():
            self.open_file(urls[0].toLocalFile())

    def closeEvent(self, event):
        if not self._confirm_discard():
            event.ignore()
            return
        self.filters_panel.shutdown()
        for t in list(self._zombies):  # espera as contas em segundo plano acabarem
            t.wait(15000)
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("state", self.saveState())
        super().closeEvent(event)
