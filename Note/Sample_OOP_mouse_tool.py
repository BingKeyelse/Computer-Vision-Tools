from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPainter, QPen, QPixmap
from PyQt5.QtCore import Qt
import sys

# ==== Base class cho tất cả tools ====
class MouseTool:
    def on_mouse_down(self, event): pass
    def on_mouse_move(self, event): pass
    def on_mouse_up(self, event): pass
    def draw(self, painter): pass

# ==== Tool cụ thể: vẽ Box ====
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

# ==== Quản lý tool ====
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

# ==== Canvas ====
class Canvas(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(800, 600)
        self.pixmap_obj = QPixmap(self.size())
        self.pixmap_obj.fill(Qt.white)
        self.setPixmap(self.pixmap_obj)
        self.tool_manager = None

    def set_tool_manager(self, manager):
        self.tool_manager = manager

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

# ==== MainWindow ====
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Tool")
        self.canvas = Canvas()
        self.setCentralWidget(self.canvas)

        self.tool_manager = ToolManager()
        self.canvas.set_tool_manager(self.tool_manager)

        # Chọn tool mặc định là Box
        self.tool_manager.set_tool(BoxTool())

# ==== Run app ====
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
