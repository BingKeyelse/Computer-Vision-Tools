import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt


# ===== Shape base class =====
class Shape:
    def draw(self, painter):
        raise NotImplementedError


class BoxShape(Shape):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def draw(self, painter):
        x1, y1 = self.start
        x2, y2 = self.end
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)


# ===== History manager =====
class History:
    def __init__(self):
        self.done = []   # stack đã thực hiện
        self.undone = [] # stack undo

    def add_action(self, shape):
        self.done.append(shape)
        self.undone.clear()

    def undo(self):
        if self.done:
            shape = self.done.pop()
            self.undone.append(shape)

    def redo(self):
        if self.undone:
            shape = self.undone.pop()
            self.done.append(shape)

    def get_all_shapes(self):
        return self.done


# ===== Tool base class =====
class MouseTool:
    def on_mouse_down(self, event): pass
    def on_mouse_move(self, event): pass
    def on_mouse_up(self, event): pass
    def draw(self, painter): pass


class BoxTool(MouseTool):
    def __init__(self, history):
        self.start = None
        self.end = None
        self.history = history
        self.temp_shape = None  # để vẽ preview khi kéo chuột

    def on_mouse_down(self, event):
        self.start = (event.x(), event.y())
        self.end = self.start
        self.temp_shape = BoxShape(self.start, self.end)

    def on_mouse_move(self, event):
        if self.start:
            self.end = (event.x(), event.y())
            self.temp_shape = BoxShape(self.start, self.end)

    def on_mouse_up(self, event):
        if self.start:
            self.end = (event.x(), event.y())
            shape = BoxShape(self.start, self.end)
            self.history.add_action(shape)
        self.start = None
        self.end = None
        self.temp_shape = None

    def draw(self, painter):
        if self.temp_shape:
            self.temp_shape.draw(painter)


# ===== Canvas =====
class Canvas(QLabel):
    def __init__(self, history, tool, parent=None):
        super().__init__(parent)
        self.setFixedSize(800, 600)
        self.history = history
        self.tool = tool

    def mousePressEvent(self, event):
        self.tool.on_mouse_down(event)
        self.update()

    def mouseMoveEvent(self, event):
        self.tool.on_mouse_move(event)
        self.update()

    def mouseReleaseEvent(self, event):
        self.tool.on_mouse_up(event)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 2))

        # vẽ tất cả shapes từ history
        for shape in self.history.get_all_shapes():
            shape.draw(painter)

        # vẽ preview (nếu đang kéo chuột)
        self.tool.draw(painter)


# ===== MainWindow =====
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Box Tool + Undo/Redo Demo")

        # tạo history và tool
        self.history = History()
        self.tool = BoxTool(self.history)

        # tạo canvas
        self.canvas = Canvas(self.history, self.tool)
        self.setCentralWidget(self.canvas)

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:  # Ctrl + ...
            if event.key() == Qt.Key_Z:  # Ctrl+Z
                self.history.undo()
                self.canvas.update()
            elif event.key() == Qt.Key_Y:  # Ctrl+Y
                self.history.redo()
                self.canvas.update()


# ===== Run app =====
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
