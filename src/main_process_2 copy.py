from libs import*

from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor
from PyQt5.QtCore import Qt, QPointF
import numpy as np
import cv2

class SimpleEvent:
    """Wrapper nhỏ cung cấp x(), y() tương tự QMouseEvent nhưng ở hệ toạ độ ảnh."""
    def __init__(self, x, y, qt_event=None):
        self._x = int(round(x))
        self._y = int(round(y))
        self.qt_event = qt_event
    def x(self): return self._x
    def y(self): return self._y
    # bạn có thể thêm button(), modifiers() nếu cần

class Canvas(QLabel):
    def __init__(self, img, parent=None):
        super().__init__(parent)
        self.tool_manager = None

        # ảnh gốc (QPixmap) và ảnh hiển thị đã scale (QPixmap)
        self.orig_image = self.cvimg_to_qpixmap(img)  # QPixmap (original pixels)
        self.display_pixmap = None  # QPixmap phù hợp kích thước widget hiện tại

        # Tắt scaledContents — ta scale thủ công
        self.setScaledContents(False)

        # nếu widget đã có size lúc tạo (thường không), scale ngay
        if self.orig_image and self.width() > 0 and self.height() > 0:
            self._update_display_pixmap()

    def set_tool_manager(self, manager):
        self.tool_manager = manager

    def cvimg_to_qpixmap(self, cv_img):
        if cv_img is None:
            return None
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _update_display_pixmap(self):
        """Scale orig_image → display_pixmap theo kích thước widget, giữ tỉ lệ."""
        if not self.orig_image:
            self.display_pixmap = None
            self.clear()
            return
        self.display_pixmap = self.orig_image.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        # Gắn cho QLabel (để nền ảnh hiển thị)
        self.setPixmap(self.display_pixmap)

    def resizeEvent(self, event):
        # mỗi khi widget thay đổi kích thước, cập nhật ảnh hiển thị
        if self.orig_image:
            self._update_display_pixmap()
        super().resizeEvent(event)

    def widget_to_image(self, pos):
        """
        Chuyển pos (QPoint hoặc pair x,y) từ toạ độ widget -> toạ độ ảnh gốc.
        Trả (img_x, img_y) float. Nếu ngoài vùng ảnh, trả None.
        """
        if self.display_pixmap is None or self.orig_image is None:
            return None

        # pos có thể là QPoint hoặc tuple
        if hasattr(pos, "x"):
            wx, wy = pos.x(), pos.y()
        else:
            wx, wy = pos

        pw, ph = self.display_pixmap.width(), self.display_pixmap.height()
        lw, lh = self.width(), self.height()

        # image có thể được letterbox: tính offset hiển thị
        xoff = int((lw - pw) / 2)
        yoff = int((lh - ph) / 2)

        ix = wx - xoff
        iy = wy - yoff

        if ix < 0 or iy < 0 or ix >= pw or iy >= ph:
            return None

        # tỉ lệ giữa ảnh gốc và display
        orig_w, orig_h = self.orig_image.width(), self.orig_image.height()
        scale_x = orig_w / pw
        scale_y = orig_h / ph
        # vì ta dùng KeepAspectRatio thì scale_x == scale_y (gần như)
        img_x = ix * scale_x
        img_y = iy * scale_y
        return (img_x, img_y)

    def paintEvent(self, event):
        # QLabel vẽ pixmap trước
        super().paintEvent(event)

        if not self.tool_manager:
            return

        # Nếu chưa có display_pixmap thì ko vẽ overlay
        if self.display_pixmap is None:
            return

        # Setup painter nhưng vẽ overlay theo hệ toạ độ ảnh (origin = top-left of displayed image)
        painter = QPainter(self)

        pw, ph = self.display_pixmap.width(), self.display_pixmap.height()
        lw, lh = self.width(), self.height()
        xoff = (lw - pw) / 2.0
        yoff = (lh - ph) / 2.0

        # scale factor từ ảnh gốc -> display
        orig_w, orig_h = self.orig_image.width(), self.orig_image.height()
        sx = pw / orig_w
        sy = ph / orig_h
        # giữ tỉ lệ nên sx ~ sy; chúng ta scale theo sx,sy (đều) hoặc sx
        # để tránh méo, dùng sx (sx==sy if KeepAspectRatio)
        painter.save()
        painter.translate(xoff, yoff)
        painter.scale(sx, sy)

        # NOW: mọi thứ vẽ bằng painter phải ở hệ toạ độ ảnh gốc
        # tool_manager.draw(painter) phải vẽ dùng toạ độ ảnh (start/end in image px)
        self.tool_manager.draw(painter)

        painter.restore()

    # ---- mouse handlers: map to image coords and forward to tool manager ----
    def mousePressEvent(self, event):
        if not self.tool_manager:
            return
        mapped = self.widget_to_image(event.pos())
        if mapped is None:
            return  # click ngoài vùng ảnh -> ignore
        ev = SimpleEvent(mapped[0], mapped[1], event)
        self.tool_manager.handle_event("mouse_down", ev)
        self.update()

    def mouseMoveEvent(self, event):
        if not self.tool_manager:
            return
        mapped = self.widget_to_image(event.pos())
        if mapped is None:
            return
        ev = SimpleEvent(mapped[0], mapped[1], event)
        self.tool_manager.handle_event("mouse_move", ev)
        self.update()

    def mouseReleaseEvent(self, event):
        if not self.tool_manager:
            return
        mapped = self.widget_to_image(event.pos())
        if mapped is None:
            return
        ev = SimpleEvent(mapped[0], mapped[1], event)
        self.tool_manager.handle_event("mouse_up", ev)
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.init_UX_UI()

        image_ori= cv2.imread(r'src\data\images\image.jpg')

        # Tạo canvas và gắn vào QLabel có sẵn trong ui
        self.canvas = Canvas(image_ori, parent=self.ui.screen_main) # Nơi để hiển thị và là nơi thao tác chính
        self.canvas.setGeometry(self.ui.label.rect())  # khớp kích thước với label

        self.tool_manager = ToolManager() # Đạo diễn, người chỉ định dùng tool nào
        self.canvas.set_tool_manager(self.tool_manager)

        # Chọn tool mặc định là Box
        self.tool_manager.set_tool(BoxTool())

        self.ui.btn_shape.currentTextChanged.connect(self.change_tool)

        self.ui.btn_cut.clicked.connect(lambda: (self.tool_manager.cut(), self.canvas.update()))
        self.ui.btn_clear.clicked.connect(lambda: (self.tool_manager.clear(), self.canvas.update()))
        self.ui.btn_undo.clicked.connect(lambda: (self.tool_manager.undo(), self.canvas.update()))
        self.ui.btn_polyundo.clicked.connect(lambda: (self.tool_manager.undo_polygon(), self.canvas.update()))
    
    def init_UX_UI(self):
        self.ui.btn_polyundo.hide()

    
    def change_tool(self, tool_name):
        self.ui.btn_polyundo.hide()
        if tool_name == "Box":
            self.tool_manager.set_tool(BoxTool())
        elif tool_name == "Circle":
            self.tool_manager.set_tool(CircleTool())
        elif tool_name == "Polygon":
            self.tool_manager.set_tool(PolygonTool())
            self.ui.btn_polyundo.show()
        elif tool_name == "Oriented Box":
            self.tool_manager.set_tool(OrientedBoxTool())
       

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())