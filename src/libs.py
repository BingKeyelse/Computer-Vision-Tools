import sys
import os

import numpy as np
import math
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt, QPoint, QPointF, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor, QPolygon, QPainterPath, QPolygonF
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QApplication, QWidget, QPushButton,\
                            QVBoxLayout, QFileDialog, QMenu, QAction, QApplication, QListWidget

from pyqt5_ui.gui import Ui_MainWindow
import cv2

# ==== Base class cho tất cả tools ====
class MouseTool:
    """
    Hàm cơ sở của toàn bộ thao tác chuột gồm:
    nhấn. di, nhả, vẽ hình, trả lại shape cho ToolManager
    """
    def on_mouse_down(self): pass
    def on_mouse_move(self, x_offset=0, y_offset=0, scale=1.0): pass
    def on_mouse_up(self): pass
    def draw(self): pass
    def reset_image(self): pass

    ## Thêm
    def get_shape(self):
        """Trả về dữ liệu hình đã vẽ xong (nếu có) gồm kiểu shape và thông số cần thiết"""
        return None

from mouse.Box import BoxTool
from mouse.Circle import CircleTool
from mouse.OrientedBox import OrientedBoxTool
from mouse.Polygon import PolygonTool
from mouse.Tools import ToolManager

from Function_button import ButtonController
