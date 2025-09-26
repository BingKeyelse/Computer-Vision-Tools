from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPainter, QPen, QPixmap
from PyQt5.QtCore import Qt

class Canvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap_obj = QPixmap(800, 600)  # ảnh nền hoặc canvas trống
        self.pixmap_obj.fill(Qt.white)
        self.setPixmap(self.pixmap_obj)

        self.tool_manager = None  # sẽ set từ MainWindow

    def set_tool_manager(self, manager):
        self.tool_manager = manager

    # mouse events
    def mousePressEvent(self, event):
        if self.tool_manager:
            self.tool_manager.handle_event("mouse_down", event)
        self.update()

    def mouseMoveEvent(self, event):
        if self.tool_manager:
            self.tool_manager.handle_event("mouse_move", event)
        self.update()

    def mouseReleaseEvent(self, event):
        if self.tool_manager:
            self.tool_manager.handle_event("mouse_up", event)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.tool_manager:
            painter = QPainter(self)
            self.tool_manager.draw(painter)

# =================
class MouseTool:
    def on_mouse_down(self, event): pass
    def on_mouse_move(self, event): pass
    def on_mouse_up(self, event): pass
    def draw(self, painter): pass

# =================
class BoxTool(MouseTool):
    def __init__(self):
        self.start = None
        self.end = None

    def on_mouse_down(self, event):
        self.start = (event.x(), event.y())
        self.end = self.start

    def on_mouse_move(self, event):
        if self.start:
            self.end = (event.x(), event.y())

    def on_mouse_up(self, event):
        self.end = (event.x(), event.y())

    def draw(self, painter):
        if self.start and self.end:
            pen = QPen(Qt.red, 2)
            painter.setPen(pen)
            x1, y1 = self.start
            x2, y2 = self.end
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

# =================
class ToolManager:
    def __init__(self):
        self.active_tool = None

    def set_tool(self, tool: MouseTool):
        self.active_tool = tool

    def handle_event(self, event_type, event):
        if not self.active_tool:
            return
        if event_type == "mouse_down":
            self.active_tool.on_mouse_down(event)
        elif event_type == "mouse_move":
            self.active_tool.on_mouse_move(event)
        elif event_type == "mouse_up":
            self.active_tool.on_mouse_up(event)

    def draw(self, painter):
        if self.active_tool:
            self.active_tool.draw(painter)

