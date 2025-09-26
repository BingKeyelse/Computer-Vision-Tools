# file: interactive_canvas.py
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout
)
from PyQt5.QtGui import QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPointF, QRectF

# ---------- Helper math / clamp ----------
def clamp(v, a, b):
    return max(a, min(b, v))

# ---------- ImageCanvas Widget ----------
class ImageCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None           # numpy BGR image (original)
        self.qimage = None          # QImage from original (RGB)
        self.annotations = []       # list of dicts: {'type':'rect'/'poly','points':[(x,y),...]}
        self.current = None         # current drawing {'type':..., 'points':[...] }
        self.mode = 'none'          # 'rect', 'poly', 'none'
        self.dragging = None        # (ann_idx, pt_idx) when dragging a control point
        self.hit_radius = 6         # px in display coords for selecting points

        # cached layout variables (updated on paint)
        self._s = 1.0
        self._ox = 0
        self._oy = 0
        self._draw_w = 0
        self._draw_h = 0

        self.setMouseTracking(True)

    def set_image(self, bgr_image):
        """Set current image (numpy BGR)."""
        if bgr_image is None:
            self.image = None
            self.qimage = None
            self.update()
            return

        self.image = bgr_image.copy()
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = rgb.strides[0]
        self.qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self.update()

    # ---------- coordinate transforms ----------
    def _compute_scale_and_offset(self):
        """Compute s, ox, oy for KeepAspectRatio mapping image->widget."""
        if self.qimage is None:
            self._s = 1.0; self._ox = 0; self._oy = 0; self._draw_w = 0; self._draw_h = 0
            return
        W_img = self.qimage.width()
        H_img = self.qimage.height()
        W_w = max(1, self.width())
        H_w = max(1, self.height())
        s = min(W_w / W_img, H_w / H_img)
        draw_w = int(W_img * s)
        draw_h = int(H_img * s)
        ox = (W_w - draw_w) / 2
        oy = (H_w - draw_h) / 2
        self._s, self._ox, self._oy, self._draw_w, self._draw_h = s, ox, oy, draw_w, draw_h

    def image_to_display(self, x_img, y_img):
        """Map image coords -> display coords (QWidget coords)."""
        x = self._ox + x_img * self._s
        y = self._oy + y_img * self._s
        return QPointF(x, y)

    def display_to_image(self, x_disp, y_disp):
        """Map display coords -> image coords (float)."""
        x_img = (x_disp - self._ox) / self._s
        y_img = (y_disp - self._oy) / self._s
        if self.qimage:
            x_img = clamp(x_img, 0, self.qimage.width() - 1)
            y_img = clamp(y_img, 0, self.qimage.height() - 1)
        return (x_img, y_img)

    # ---------- painting ----------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self.qimage is None:
            return

        self._compute_scale_and_offset()
        target_rect = QRectF(self._ox, self._oy, self._draw_w, self._draw_h)
        source_rect = QRectF(0, 0, self.qimage.width(), self.qimage.height())
        painter.drawImage(target_rect, self.qimage, source_rect)

        # draw annotations (mapped to display coords)
        pen_ann = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen_ann)
        painter.setRenderHint(QPainter.Antialiasing)

        for ann in self.annotations:
            if ann['type'] == 'rect':
                (x1, y1), (x2, y2) = ann['points']
                p1 = self.image_to_display(x1, y1)
                p2 = self.image_to_display(x2, y2)
                rect = QRectF(p1, p2)
                painter.drawRect(rect.normalized())
                # draw handles
                self._draw_handle(painter, p1); self._draw_handle(painter, p2)

            elif ann['type'] == 'poly':
                pts = [self.image_to_display(x, y) for (x, y) in ann['points']]
                if len(pts) >= 2:
                    painter.drawPolygon(*pts)
                for p in pts:
                    self._draw_handle(painter, p)

        # draw current drawing in different color
        if self.current:
            pen_cur = QPen(QColor(255, 165, 0), 2)  # orange
            painter.setPen(pen_cur)
            if self.current['type'] == 'rect' and len(self.current['points']) == 2:
                p1 = self.image_to_display(*self.current['points'][0])
                p2 = self.image_to_display(*self.current['points'][1])
                painter.drawRect(QRectF(p1, p2).normalized())
            elif self.current['type'] == 'poly':
                pts = [self.image_to_display(x, y) for (x, y) in self.current['points']]
                if len(pts) >= 2:
                    painter.drawPolyline(*pts)
                for p in pts:
                    self._draw_handle(painter, p)

    def _draw_handle(self, painter, qp_point, size=6):
        r = size
        painter.save()
        painter.setBrush(QColor(0, 180, 255))
        painter.drawEllipse(QRectF(qp_point.x()-r/2, qp_point.y()-r/2, r, r))
        painter.restore()

    # ---------- mouse interaction ----------
    def mousePressEvent(self, event):
        x, y = event.x(), event.y()
        if self.qimage is None:
            return

        # only react if clicking inside image area
        if not (self._ox <= x <= self._ox + self._draw_w and self._oy <= y <= self._oy + self._draw_h):
            return

        # 1) check if clicked close to an existing handle (for dragging)
        found = self._find_hit_handle(x, y)
        if found:
            self.dragging = found   # (ann_idx, pt_idx)
            return

        # 2) otherwise, start drawing depending on mode
        ix, iy = self.display_to_image(x, y)
        if self.mode == 'rect':
            # start rect: first click = start, drag to create
            self.current = {'type': 'rect', 'points': [(ix, iy), (ix, iy)]}
        elif self.mode == 'poly':
            # polygon: each click adds a vertex; right-click finish
            if event.button() == Qt.RightButton:
                # finish polygon
                if self.current and self.current['type'] == 'poly' and len(self.current['points']) >= 3:
                    self.annotations.append(self.current)
                self.current = None
            else:
                if self.current is None:
                    self.current = {'type': 'poly', 'points': [(ix, iy)]}
                else:
                    self.current['points'].append((ix, iy))
        self.update()

    def mouseMoveEvent(self, event):
        x, y = event.x(), event.y()
        if self.qimage is None:
            return
        if self.dragging is not None:
            # update dragged control point (modify annotation in image coords)
            ann_idx, pt_idx = self.dragging
            ix, iy = self.display_to_image(x, y)
            ann = self.annotations[ann_idx]
            ann['points'][pt_idx] = (ix, iy)
            self.update()
            return

        if self.current and self.current['type'] == 'rect':
            # update rect current endpoint
            ix, iy = self.display_to_image(x, y)
            self.current['points'][1] = (ix, iy)
            self.update()

    def mouseReleaseEvent(self, event):
        if self.dragging is not None:
            self.dragging = None
            return

        if self.current and self.current['type'] == 'rect':
            # finalize rect (normalize points)
            (x1, y1), (x2, y2) = self.current['points']
            if abs(x2 - x1) > 2 and abs(y2 - y1) > 2:
                rect = {'type':'rect', 'points': [(x1, y1), (x2, y2)]}
                self.annotations.append(rect)
            self.current = None
            self.update()

    def _find_hit_handle(self, x_disp, y_disp):
        """Return (ann_idx, pt_idx) if a handle is within hit_radius in display coords, else None."""
        for ai, ann in enumerate(self.annotations):
            pts = ann['points']
            for pi, (x_img, y_img) in enumerate(pts):
                p_disp = self.image_to_display(x_img, y_img)
                dx = p_disp.x() - x_disp
                dy = p_disp.y() - y_disp
                if (dx*dx + dy*dy) <= (self.hit_radius ** 2):
                    return (ai, pi)
        return None

    # ---------- helpers for save / compute area ----------
    def save_image_with_annotations(self, filename):
        """Draw annotations on a copy of original image and save with OpenCV."""
        if self.image is None:
            return False
        out = self.image.copy()
        for ann in self.annotations:
            if ann['type'] == 'rect':
                (x1, y1), (x2, y2) = ann['points']
                x1, x2 = int(round(x1)), int(round(x2))
                y1, y2 = int(round(y1)), int(round(y2))
                cv2.rectangle(out, (x1, y1), (x2, y2), (0,255,0), 2)
            elif ann['type'] == 'poly':
                pts = np.array([[int(round(x)), int(round(y))] for (x,y) in ann['points']], dtype=np.int32)
                if pts.shape[0] >= 2:
                    cv2.polylines(out, [pts], isClosed=True, color=(0,255,0), thickness=2)
        cv2.imwrite(filename, out)
        return True

    def compute_polygon_area(self, ann_idx):
        """Return area (pixels^2) for polygon annotation index."""
        ann = self.annotations[ann_idx]
        if ann['type'] != 'poly':
            return 0.0
        pts = np.array([[float(x), float(y)] for (x,y) in ann['points']], dtype=np.float32)
        return abs(cv2.contourArea(pts))

# ---------- MainWindow to demo ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Annotation Demo")
        self.canvas = ImageCanvas()

        # sample buttons
        btn_rect = QPushButton("Mode: Rect")
        btn_poly = QPushButton("Mode: Poly")
        btn_none = QPushButton("Mode: None")
        btn_clear = QPushButton("Clear")
        btn_save = QPushButton("Save Annotated Image")

        btn_rect.clicked.connect(lambda: self.set_mode('rect'))
        btn_poly.clicked.connect(lambda: self.set_mode('poly'))
        btn_none.clicked.connect(lambda: self.set_mode('none'))
        btn_clear.clicked.connect(self.clear_all)
        btn_save.clicked.connect(self.save_out)

        hl = QHBoxLayout()
        hl.addWidget(btn_rect); hl.addWidget(btn_poly); hl.addWidget(btn_none)
        hl.addWidget(btn_clear); hl.addWidget(btn_save)

        v = QVBoxLayout()
        v.addLayout(hl)
        v.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(v)
        self.setCentralWidget(container)

        # load demo image (or replace with camera frame periodically)
        img = cv2.imread('picture1.jpg')   # change path to a test image on your disk
        if img is None:
            # create synthetic image
            img = np.full((480,640,3), 200, dtype=np.uint8)
            cv2.putText(img, "No image file, synthetic", (10,240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        self.canvas.set_image(img)

    def set_mode(self, m):
        self.canvas.mode = m

    def clear_all(self):
        self.canvas.annotations = []
        self.canvas.current = None
        self.canvas.update()

    def save_out(self):
        ok = self.canvas.save_image_with_annotations("annotated_out.jpg")
        print("Saved:", ok)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(900, 600)
    w.show()
    sys.exit(app.exec_())
