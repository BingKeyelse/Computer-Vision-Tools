import sys
import os

import numpy as np
import math
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt, QPoint, QPointF, QTimer, QObject, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor, QPolygon, QPainterPath, QPolygonF
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QApplication, QWidget, QPushButton,\
                            QVBoxLayout, QFileDialog, QMenu, QAction, QApplication, QListWidget, QMessageBox, QComboBox

from pyqt5_ui.gui import Ui_MainWindow
import cv2

import glob
from abc import ABC, abstractmethod
from typing import Union
from pypylon import pylon
import time
import sqlite3

# ==== Base Mouse cho tất cả tools ====
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
    

# ==== Base Camera cho tất cả tools ====
class BaseCamera(ABC):
    """
    ## Base Camera
    """
    def __init__(self, name):
        self.name = name
        self.cap = None
        self.connected = False

    @abstractmethod 
    def connect(self):
        """
        Hàm connect: Request Override
        """
        pass

    @abstractmethod 
    def get_frame(self):
        """
        Hàm get frame: Request Override
        """
        pass

    def disconnect(self):
        """
        Hàm Disconect: Unrequest Override
        """
        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False

# ==== Base Matcher cho tất cả Tools Matching ====
class BaseMatcher(ABC):
    def __init__(self, temple_path: str, data: tuple, scale: float):
        """
        ## Base Matcher cho tất cả Tools Matching
        - Args:
            - image_path: link ảnh
            - data: dữ liệu truyền vào
        """
        self.temple_path = temple_path
        self.data = data  # có thể là box, circle, polygon...
        self.scale = scale
        self.template = None

    @abstractmethod
    def load_template(self):
        """## Load ảnh temple để matching với ảnh gốc"""
        pass

    @abstractmethod
    def match(self, scene):
        """Trả về dict: { 'type': ..., 'box': ..., 'score': ..., 'angle': ... }"""
        pass

# Database
from Function_data.DataBase_Manager import DatabaseController

from Function_mouse.Box import BoxTool
from Function_mouse.Circle import CircleTool
from Function_mouse.OrientedBox import OrientedBoxTool
from Function_mouse.Polygon import PolygonTool
from Function_mouse.Tools import ToolManager

from Function_button.Button_Manager import ButtonController
from Function_button.Button_Camera import CameraFunctions 

# Camera
from Function_camera.Camera_USB import USBCamera
from Function_camera.Camera_Basler import BaslerCamera
from Function_camera.Camera_Manager import CreateNameCamera

# Matching
from Function_maching.Matching_Box import BoxMatcher
from Function_maching.Matching_OrientedBox import OrientedBoxMatcher
from Function_maching.Matching_Circle import CircleMatcher
from Function_maching.Matching_Polygon import PolygonMatcher

from Function_maching.Matching_Manager import MatcherFactory




