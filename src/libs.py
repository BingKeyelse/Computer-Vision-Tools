import sys
import cv2
import numpy as np
import math
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt, QPoint, QPointF, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor, QPolygon, QPainterPath, QPolygonF
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QApplication, QWidget, QPushButton, QVBoxLayout

from pyqt5_ui.gui import Ui_MainWindow

# ==== Base class cho tất cả tools ====
class MouseTool:
    def on_mouse_down(self, event, _offset=0, y_offset=0): pass
    def on_mouse_move(self, event, x_offset=0, y_offset=0, scaled=0.0): pass
    def on_mouse_up(self, event, _offset=0, y_offset=0): pass
    def draw(self, painter, x_offset=0, y_offset=0): pass

    ## Thêm
    def get_shape(self, x_offset=0, y_offset=0):
        """Trả về dữ liệu hình đã vẽ xong (nếu có)"""
        return None

from mouse.Box import BoxTool
from mouse.Circle import CircleTool
from mouse.OrientedBox import OrientedBoxTool
from mouse.Polygon import PolygonTool
from mouse.Tools import ToolManager
