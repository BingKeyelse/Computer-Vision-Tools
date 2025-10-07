import sys
import os

import numpy as np
import math
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt, QPoint, QPointF, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor, QPolygon, QPainterPath, QPolygonF
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QApplication, QWidget, QPushButton,\
                            QVBoxLayout, QFileDialog, QMenu, QAction, QApplication, QListWidget, QMessageBox, QComboBox

from pyqt5_ui.gui import Ui_MainWindow
import cv2

import glob
from pypylon import pylon
from abc import ABC, abstractmethod

# ==== Base class cho tất cả tools ====
class MouseTool:
    """
    Hàm cơ sở của toàn bộ thao tác chuột gồm:
    nhấn. di, nhả, vẽ hình, trả lại shape cho ToolManager
    """
    def on_mouse_down(self): 
        """Khi nhấn chuột chuột tạo ra giá trị vị trí liên quan"""
        pass
    def on_mouse_move(self): 
        """Khi di chuột chuột tạo ra giá trị vị trí liên quan"""
        pass
    
    def on_mouse_up(self): 
        """Khi thả chuột tạo ra giá trị vị trí liên quan"""
        pass
    def draw(self): 
        """Vẽ hình đang dùng để thao tác với giá trị mà chuột tương tác"""
        pass
    def reset_image(self):
        """Khởi tạo lại giá trị bắt đầu để reset hình đang vẽ
        - start, end= None
        """
        pass

    def get_shape(self):
        """Trả về dữ liệu hình đã vẽ xong (nếu có) gồm kiểu shape và thông số cần thiết"""
        return None

from Function_mouse.Box import BoxTool
from Function_mouse.Circle import CircleTool
from Function_mouse.OrientedBox import OrientedBoxTool
from Function_mouse.Polygon import PolygonTool
from Function_mouse.Tools import ToolManager

from Function_button.Button_Manager import ButtonController

#Ken
from camera.function_cam import CameraFunctions 
from camera.initial_cam import CreateNameCamera
#Ken
