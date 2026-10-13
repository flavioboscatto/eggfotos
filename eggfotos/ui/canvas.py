"""Área de visualização da foto: zoom, arrastar, seleção de recorte,
pincel e seleção de região.

As coordenadas da cena são SEMPRE os pixels da imagem em resolução total.
Prévias calculadas sobre uma versão reduzida são exibidas com escala
(set_preview), então as ferramentas continuam devolvendo pixels reais.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QFrame, QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsScene, QGraphicsView,
)

from ..core.blur import stroke_mask
from .qt_utils import to_qpixmap
from .styles import ACCENT

TOOL_NAV = "navegar"
TOOL_CROP = "recortar"
TOOL_BRUSH = "pincel"
TOOL_REGION = "regiao"

ZOOM_MIN, ZOOM_MAX = 0.02, 32.0


class ImageCanvas(QGraphicsView):
    zoomChanged = pyqtSignal(float)
    cropChanged = pyqtSignal(object)        # (x, y, w, h) em pixels ou None
    cropAccepted = pyqtSignal()             # Enter na ferramenta de recorte
    brushStrokeFinished = pyqtSignal(object)  # máscara np.uint8 HxW
    regionSelected = pyqtSignal(object)     # (x, y, w, h) em pixels

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        # xadrez atrás da foto: só aparece onde há transparência
        self._checker_item = QGraphicsRectItem()
        self._checker_item.setPen(QPen(Qt.PenStyle.NoPen))
        self._checker_item.setBrush(QBrush(self._checker_pixmap()))
        self._checker_item.setZValue(-1)
        self._scene.addItem(self._checker_item)
        self._scene.addItem(self._pixmap_item)
        self._image_size: tuple[int, int] | None = None  # (w, h)

        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#18181b"))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(False)
        self.viewport().setAcceptDrops(False)

        self._tool = TOOL_NAV
        self._space_pan = False
        self._drag_start: QPointF | None = None

        # recorte
        self._crop_ratio: float | None = None
        self._crop_rect: QRectF | None = None
        self._crop_moving = False
        self._crop_offset = QPointF()
        self._crop_item: QGraphicsRectItem | None = None
        self._shade_item: QGraphicsPathItem | None = None

        # pincel
        self.brush_size = 60
        self._stroke_points: list[tuple[float, float]] = []
        self._stroke_path: QPainterPath | None = None
        self._stroke_item: QGraphicsPathItem | None = None
        self._cursor_item: QGraphicsEllipseItem | None = None

        # região
        self.region_shape = "retangulo"
        self._region_item = None

        self._apply_drag_mode()

    @staticmethod
    def _checker_pixmap(cell: int = 12) -> QPixmap:
        pm = QPixmap(cell * 2, cell * 2)
        pm.fill(QColor("#cfcfd4"))
        p = QPainter(pm)
        p.fillRect(0, 0, cell, cell, QColor("#9a9aa2"))
        p.fillRect(cell, cell, cell, cell, QColor("#9a9aa2"))
        p.end()
        return pm

    # ================================================================ imagem
    @property
    def has_image(self) -> bool:
        return self._image_size is not None

    def set_image(self, arr: np.ndarray, reset_view: bool = False) -> None:
        """Mostra a imagem em resolução total."""
        h, w = arr.shape[:2]
        self._pixmap_item.setPixmap(to_qpixmap(arr))
        self._pixmap_item.setScale(1.0)
        self._pixmap_item.setPos(0, 0)
        size_changed = self._image_size != (w, h)
        self._image_size = (w, h)
        self._scene.setSceneRect(QRectF(0, 0, w, h))
        self._checker_item.setRect(QRectF(0, 0, w, h))
        if size_changed:
            self.clear_crop()
        if reset_view or size_changed:
            self.fit_in_view()

    def set_preview(self, arr: np.ndarray, scale: float, centered: bool = False) -> None:
        """Mostra uma prévia calculada numa versão reduzida.

        scale: fator que leva os pixels da prévia aos da imagem total.
        centered: centraliza (para resultados menores, ex.: endireitar).
        """
        if not self.has_image:
            return
        self._pixmap_item.setPixmap(to_qpixmap(arr))
        self._pixmap_item.setScale(scale)
        if centered:
            W, H = self._image_size
            w, h = arr.shape[1] * scale, arr.shape[0] * scale
            self._pixmap_item.setPos((W - w) / 2, (H - h) / 2)
        else:
            self._pixmap_item.setPos(0, 0)

    def clear(self) -> None:
        self._pixmap_item.setPixmap(to_qpixmap(np.zeros((1, 1, 3), np.uint8)))
        self._image_size = None
        self.clear_crop()

    # ================================================================== zoom
    def current_zoom(self) -> float:
        return self.transform().m11()

    def fit_in_view(self) -> None:
        if not self.has_image:
            return
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        if self.current_zoom() > 1.0:  # foto pequena: não amplia além de 100%
            self.resetTransform()
        self.zoomChanged.emit(self.current_zoom())

    def zoom_100(self) -> None:
        self.resetTransform()
        self.zoomChanged.emit(1.0)

    def zoom_by(self, factor: float) -> None:
        z = self.current_zoom()
        factor = max(ZOOM_MIN / z, min(ZOOM_MAX / z, factor))
        self.scale(factor, factor)
        self.zoomChanged.emit(self.current_zoom())

    def wheelEvent(self, event):
        if not self.has_image:
            return
        delta = event.angleDelta().y()
        if delta:
            self.zoom_by(1.25 if delta > 0 else 0.8)
        event.accept()

    # ============================================================ ferramentas
    def set_tool(self, tool: str) -> None:
        if tool == self._tool:
            return
        if self._tool == TOOL_CROP:
            self.clear_crop()
        self._remove_stroke()
        self._remove_region()
        self._hide_cursor()
        self._tool = tool
        self._apply_drag_mode()

    def tool(self) -> str:
        return self._tool

    def _apply_drag_mode(self) -> None:
        if self._tool == TOOL_NAV or self._space_pan:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().unsetCursor()
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            cursor = Qt.CursorShape.BlankCursor if self._tool == TOOL_BRUSH else Qt.CursorShape.CrossCursor
            self.viewport().setCursor(cursor)

    def set_brush_size(self, size: int) -> None:
        self.brush_size = max(1, int(size))
        if self._cursor_item is not None and self._cursor_item.isVisible():
            c = self._cursor_item.rect().center()
            self._place_cursor(c)

    def set_crop_ratio(self, ratio: float | None) -> None:
        self._crop_ratio = ratio
        if self._crop_rect is not None and ratio:
            # reajusta a seleção existente à nova proporção, mantendo o centro
            r = self._crop_rect
            w, h = r.width(), r.height()
            if w / max(h, 1e-6) > ratio:
                w = h * ratio
            else:
                h = w / ratio
            c = r.center()
            self._set_crop_rect(self._fit_inside(QRectF(c.x() - w / 2, c.y() - h / 2, w, h)))

    # ============================================================ utilitários
    def _scene_pos(self, event) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    def _clamp(self, p: QPointF) -> QPointF:
        W, H = self._image_size
        return QPointF(min(max(p.x(), 0.0), W), min(max(p.y(), 0.0), H))

    def _fit_inside(self, r: QRectF) -> QRectF:
        """Desloca o retângulo para dentro da imagem sem mudar o tamanho."""
        W, H = self._image_size
        w, h = min(r.width(), W), min(r.height(), H)
        x = min(max(r.x(), 0.0), W - w)
        y = min(max(r.y(), 0.0), H - h)
        return QRectF(x, y, w, h)

    def _rect_from_drag(self, start: QPointF, end: QPointF, ratio: float | None) -> QRectF:
        end = self._clamp(end)
        dx, dy = end.x() - start.x(), end.y() - start.y()
        w, h = abs(dx), abs(dy)
        if ratio and w > 0 and h > 0:
            # encolher (nunca crescer) mantém o retângulo dentro da imagem
            if w / h > ratio:
                w = h * ratio
            else:
                h = w / ratio
        x = start.x() if dx >= 0 else start.x() - w
        y = start.y() if dy >= 0 else start.y() - h
        return QRectF(x, y, w, h)

    @staticmethod
    def _cosmetic_pen(color: str, width: float = 1.5, dashed: bool = False) -> QPen:
        pen = QPen(QColor(color))
        pen.setWidthF(width)
        pen.setCosmetic(True)  # largura em pixels de tela, independe do zoom
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        return pen

    # ============================================================== recorte
    def crop_rect(self) -> tuple[int, int, int, int] | None:
        r = self._crop_rect
        if r is None or r.width() < 2 or r.height() < 2:
            return None
        return (int(round(r.x())), int(round(r.y())), int(round(r.width())), int(round(r.height())))

    def clear_crop(self) -> None:
        for item in (self._crop_item, self._shade_item):
            if item is not None:
                self._scene.removeItem(item)
        self._crop_item = self._shade_item = None
        self._crop_rect = None
        self._crop_moving = False
        self.cropChanged.emit(None)

    def _set_crop_rect(self, r: QRectF) -> None:
        self._crop_rect = r.normalized()
        if self._crop_item is None:
            self._shade_item = QGraphicsPathItem()
            self._shade_item.setBrush(QBrush(QColor(0, 0, 0, 150)))
            self._shade_item.setPen(QPen(Qt.PenStyle.NoPen))
            self._shade_item.setZValue(10)
            self._scene.addItem(self._shade_item)
            self._crop_item = QGraphicsRectItem()
            self._crop_item.setPen(self._cosmetic_pen(ACCENT, 1.5, dashed=True))
            self._crop_item.setZValue(11)
            self._scene.addItem(self._crop_item)
        self._crop_item.setRect(self._crop_rect)
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.OddEvenFill)
        path.addRect(self._scene.sceneRect())
        path.addRect(self._crop_rect)
        self._shade_item.setPath(path)
        self.cropChanged.emit(self.crop_rect())

    # ================================================================ pincel
    def _place_cursor(self, p: QPointF) -> None:
        if self._cursor_item is None:
            self._cursor_item = QGraphicsEllipseItem()
            self._cursor_item.setPen(self._cosmetic_pen("#ffffff", 1.2))
            self._cursor_item.setZValue(20)
            self._scene.addItem(self._cursor_item)
        r = self.brush_size / 2
        self._cursor_item.setRect(QRectF(p.x() - r, p.y() - r, 2 * r, 2 * r))
        self._cursor_item.setVisible(True)

    def _hide_cursor(self) -> None:
        if self._cursor_item is not None:
            self._cursor_item.setVisible(False)

    def _remove_stroke(self) -> None:
        if self._stroke_item is not None:
            self._scene.removeItem(self._stroke_item)
        self._stroke_item = None
        self._stroke_path = None
        self._stroke_points = []

    # ================================================================ região
    def _remove_region(self) -> None:
        if self._region_item is not None:
            self._scene.removeItem(self._region_item)
        self._region_item = None

    def _update_region(self, r: QRectF) -> None:
        if self._region_item is None:
            cls = QGraphicsEllipseItem if self.region_shape == "elipse" else QGraphicsRectItem
            self._region_item = cls()
            self._region_item.setPen(self._cosmetic_pen(ACCENT, 1.5, dashed=True))
            self._region_item.setBrush(QBrush(QColor(242, 177, 52, 50)))
            self._region_item.setZValue(12)
            self._scene.addItem(self._region_item)
        self._region_item.setRect(r)

    # ================================================================ eventos
    def mousePressEvent(self, event):
        self.setFocus()
        if (self._space_pan or self._tool == TOOL_NAV or not self.has_image
                or event.button() != Qt.MouseButton.LeftButton):
            return super().mousePressEvent(event)
        p = self._clamp(self._scene_pos(event))
        if self._tool == TOOL_CROP:
            if self._crop_rect is not None and self._crop_rect.contains(p):
                self._crop_moving = True
                self._crop_offset = p - self._crop_rect.topLeft()
            else:
                self._drag_start = p
                self._set_crop_rect(QRectF(p, p))
        elif self._tool == TOOL_BRUSH:
            self._remove_stroke()
            self._stroke_points = [(p.x(), p.y())]
            self._stroke_path = QPainterPath(p)
            self._stroke_path.lineTo(p + QPointF(0.01, 0.01))
            self._stroke_item = QGraphicsPathItem(self._stroke_path)
            pen = QPen(QColor(255, 90, 90, 120))
            pen.setWidthF(self.brush_size)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            self._stroke_item.setPen(pen)
            self._stroke_item.setZValue(15)
            self._scene.addItem(self._stroke_item)
        elif self._tool == TOOL_REGION:
            self._remove_region()
            self._drag_start = p
        event.accept()

    def mouseMoveEvent(self, event):
        if not self.has_image or self._space_pan or self._tool == TOOL_NAV:
            return super().mouseMoveEvent(event)
        raw = self._scene_pos(event)
        if self._tool == TOOL_BRUSH:
            self._place_cursor(raw)
            if self._stroke_path is not None:
                p = self._clamp(raw)
                self._stroke_points.append((p.x(), p.y()))
                self._stroke_path.lineTo(p)
                self._stroke_item.setPath(self._stroke_path)
        elif self._tool == TOOL_CROP:
            if self._crop_moving and self._crop_rect is not None:
                r = QRectF(self._crop_rect)
                r.moveTopLeft(raw - self._crop_offset)
                self._set_crop_rect(self._fit_inside(r))
            elif self._drag_start is not None:
                self._set_crop_rect(self._rect_from_drag(self._drag_start, raw, self._crop_ratio))
        elif self._tool == TOOL_REGION and self._drag_start is not None:
            self._update_region(self._rect_from_drag(self._drag_start, raw, None))
        event.accept()

    def mouseReleaseEvent(self, event):
        if (not self.has_image or self._space_pan or self._tool == TOOL_NAV
                or event.button() != Qt.MouseButton.LeftButton):
            return super().mouseReleaseEvent(event)
        if self._tool == TOOL_CROP:
            self._drag_start = None
            self._crop_moving = False
            if self.crop_rect() is None:
                self.clear_crop()
        elif self._tool == TOOL_BRUSH and self._stroke_points:
            W, H = self._image_size
            mask = stroke_mask((H, W), self._stroke_points, self.brush_size)
            self._remove_stroke()
            self.brushStrokeFinished.emit(mask)
        elif self._tool == TOOL_REGION and self._drag_start is not None:
            r = self._region_item.rect() if self._region_item is not None else QRectF()
            self._drag_start = None
            self._remove_region()
            if r.width() >= 2 and r.height() >= 2:
                self.regionSelected.emit((int(r.x()), int(r.y()), int(r.width()), int(r.height())))
        event.accept()

    def leaveEvent(self, event):
        self._hide_cursor()
        super().leaveEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan = True  # segurar espaço = mão para arrastar
            self._hide_cursor()
            self._apply_drag_mode()
            return
        if self._tool == TOOL_CROP and key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.cropAccepted.emit()
            return
        if key == Qt.Key.Key_Escape:
            self.clear_crop()
            self._remove_stroke()
            self._remove_region()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan = False
            self._apply_drag_mode()
            return
        super().keyReleaseEvent(event)
