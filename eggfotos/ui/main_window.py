"""Janela principal do EggFotos."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import cv2
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QDockWidget, QFileDialog, QInputDialog, QLabel, QMainWindow, QMessageBox,
    QProgressDialog, QStyle, QTabWidget, QToolButton,
)

from .. import APP_NAME, __version__
from ..core import background, faces, geometry, inpaint
from ..core.blur import blur_with_mask
from ..core.history import History
from ..core.image_io import OPEN_FILTER, is_supported, load_image, save_image
from ..pipeline.operations import apply_step, get_operation
from ..pipeline.recipe import Recipe, recipes_dir, safe_filename
from .batch_dialog import BatchDialog
from .canvas import TOOL_BRUSH, TOOL_CROP, TOOL_NAV, TOOL_REGION, ImageCanvas
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
        self._wm_strokes = 0
        self._wm_brush = False                    # houve traço de pincel (marcação não reproduzível)
        self._task: TaskWorker | None = None
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
        self.tabs = QTabWidget()
        self.tabs.addTab(self.adjust_panel, "Ajustes")
        self.tabs.addTab(self.filters_panel, "Filtros")
        self.tabs.addTab(self.crop_panel, "Recortar")
        self.tabs.addTab(self.blur_panel, "Borrar")
        self.tabs.addTab(self.background_panel, "Fundo")
        self.tabs.addTab(self.faces_panel, "Ocultar rosto")
        self.tabs.addTab(self.wm_panel, "Marca d'água")
        dock = QDockWidget("Ferramentas", self)
        dock.setObjectName("dock_tools")
        dock.setWidget(self.tabs)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        dock.setMinimumWidth(360)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

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
        self.act_zoom_in = self._action("Aumentar zoom", lambda: self.canvas.zoom_by(1.25), QKeySequence.StandardKey.ZoomIn)
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
        self.act_undo.setEnabled(self.history.can_undo())
        self.act_redo.setEnabled(self.history.can_redo())
        self.act_undo.setText(f"Desfazer {self.history.undo_label()}".strip())
        self.act_redo.setText(f"Refazer {self.history.redo_label()}".strip())
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
        """Roda uma operação na resolução total com cursor de espera.
        (Fase seguinte: mover para QThread com barra de progresso.)"""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            return func()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, f"Não foi possível aplicar:\n{e}")
            return None
        finally:
            QApplication.restoreOverrideCursor()

    def _apply_op(self, step: dict, label: str | None = None) -> None:
        """Aplica um step reproduzível à foto inteira e grava no histórico."""
        if self.image is None:
            return
        out = self._busy(lambda: apply_step(self.image, step))
        if out is not None:
            self.commit(out, label or get_operation(step["op"]).label, step)

    # ============================================================= prévias
    def _run_preview(self, render: Callable[[], None]) -> None:
        if self.image is None:
            return
        self._uncheck_original()
        self._pending = render
        try:
            render()
        except Exception as e:  # noqa: BLE001
            self.statusBar().showMessage(f"Erro na prévia: {e}", 4000)

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
        out = self._busy(lambda: geometry.crop(self.image, x, y, w, h))  # pixels exatos
        if out is not None:
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
            out = self._busy(lambda: geometry.resize(self.image, nw, nh))
            if out is not None:
                self.commit(out, f"Redimensionar {nw}×{nh}", None)

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
        out = self._busy(lambda: blur_with_mask(self.image, mask, bp.tipo(), bp.intensidade(), bp.cor()))
        if out is not None:
            self.commit(out, "Borrar (pincel)", None)

    def _apply_region_blur(self, rect) -> None:
        if self.image is None:
            return
        if self.tabs.currentIndex() == TAB_WATERMARK:
            H, W = self.image.shape[:2]
            x, y, w, h = rect
            self._wm_rects.append((x / W, y / H, w / W, h / H))
            self._wm_add_mask(inpaint.rect_mask(self.image.shape, x / W, y / H, w / W, h / H), brush=False)
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
        self.background_panel.reset()

    def _run_task(self, text: str, func: Callable[[], object], on_done: Callable[[object], None]) -> None:
        """Roda `func` numa QThread com uma janela de espera (sem barra de progresso real)."""
        if self._task is not None:
            return
        dlg = QProgressDialog(text, "", 0, 0, self)
        dlg.setCancelButton(None)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setWindowTitle(APP_NAME)
        dlg.show()
        task = TaskWorker(func, self)

        def finish():
            dlg.close()
            task.deleteLater()
            self._task = self._task_dialog = None

        def ok(result):
            finish()
            on_done(result)

        def fail(msg):
            finish()
            QMessageBox.warning(self, APP_NAME, msg)

        task.done.connect(ok)
        task.failed.connect(fail)
        self._task, self._task_dialog = task, dlg
        task.start()

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
            self._proxy_matte = None
            self.background_panel.set_detected(True)
            self._preview_background(self.background_panel.values())

        self._run_task("Detectando o objeto…", lambda: background.compute_matte(img, modelo), done)

    def _matte_ready(self, modelo: str) -> bool:
        return (self._matte is not None
                and self._matte_key == (self.history.current_uid, modelo))

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
        step = self._background_step(values)
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

    # ======================================================== marca d'água
    def _wm_reset(self) -> None:
        self._wm_mask = None
        self._wm_rects = []
        self._wm_brush = False
        self._wm_strokes = 0
        self.wm_panel.set_marked(0)

    def _wm_count(self) -> int:
        return len(self._wm_rects) + self._wm_strokes

    def _sync_wm_tool(self) -> None:
        if self.tabs.currentIndex() != TAB_WATERMARK:
            return
        if self.wm_panel.tool() == "pincel":
            self.canvas.set_tool(TOOL_BRUSH)
            self.canvas.set_brush_size(self.wm_panel.brush_size())
        else:
            self.canvas.region_shape = "retangulo"
            self.canvas.set_tool(TOOL_REGION)

    def _wm_add_mask(self, mask: np.ndarray, brush: bool) -> None:
        self._wm_mask = mask.copy() if self._wm_mask is None else np.maximum(self._wm_mask, mask)
        if brush:
            self._wm_brush = True
            self._wm_strokes += 1
        self.wm_panel.set_marked(self._wm_count())
        self._wm_show()

    def _wm_show(self) -> None:
        """Mostra a marcação em vermelho translúcido sobre a foto."""
        if self._wm_mask is None or self.image is None:
            return
        mask = self._wm_mask

        def render():
            ph, pw = self.proxy.shape[:2]
            m = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST) > 0
            out = self.proxy.copy()
            red = np.array([255, 40, 40], np.float32)
            out[m] = (out[m][..., :3] * 0.4 + red * 0.6).astype(np.uint8) if out.shape[2] == 3 else out[m]
            self.canvas.set_preview(out, 1.0 / self.proxy_scale)
        self._run_preview(render)

    def _wm_clear(self) -> None:
        self._wm_reset()
        self._cancel_preview()

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
        if not self._wm_brush and len(self._wm_rects) == 1:
            x, y, w, h = self._wm_rects[0]
            step = get_operation("remover_marca_regiao").new_step(x=x, y=y, w=w, h=h, **values)
        self._run_task("Removendo a marca d'água…",
                       lambda: inpaint.remove_with_mask(img, mask, **values),
                       lambda out: self.commit(out, "Remover marca d'água", step))


    # ========================================================= histórico
    def undo(self) -> None:
        if self.history.can_undo():
            label = self.history.undo_label()
            self._set_working_image(self.history.undo())
            self.statusBar().showMessage(f"Desfeito: {label}", 2500)

    def redo(self) -> None:
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
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Abrir foto", self._last_dir(), OPEN_FILTER)
        if path:
            self.open_file(path)

    def open_file(self, path: str | Path) -> None:
        path = Path(path)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            img, meta = load_image(path)
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, APP_NAME, f"Não foi possível abrir {path.name}:\n{e}")
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
        if self.image is None:
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
        if self.image is None:
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
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)
