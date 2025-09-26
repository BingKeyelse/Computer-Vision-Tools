# import sys
# import cv2
# import time
# from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
# from PyQt5.QtCore import QTimer
# from PyQt5.QtGui import QImage, QPixmap

# class CameraApp(QMainWindow):
#     def __init__(self):
#         super().__init__()

#         self.setWindowTitle("Realtime Camera with Trigger")
#         self.setGeometry(100, 100, 800, 600)

#         # Camera
#         self.cap = cv2.VideoCapture(0)
#         self.current_frame = None  # frame hiện tại

#         # Giao diện
#         self.label = QLabel(self)
#         self.label.setFixedSize(640, 480)

#         self.btn_trigger = QPushButton("Trig-Soft", self)
#         self.btn_trigger.clicked.connect(self.save_image)

#         layout = QVBoxLayout()
#         layout.addWidget(self.label)
#         layout.addWidget(self.btn_trigger)

#         container = QWidget()
#         container.setLayout(layout)
#         self.setCentralWidget(container)

#         # Timer update camera
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_frame)
#         self.timer.start(30)  # ~30 fps

#     def update_frame(self):
#         ret, frame = self.cap.read()
#         if ret:
#             frame = cv2.flip(frame, 1)  
#             self.current_frame = frame.copy()  # lưu lại frame hiện tại
#             rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             h, w, ch = rgb_image.shape
#             bytes_per_line = ch * w
#             qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
#             self.label.setPixmap(QPixmap.fromImage(qt_image))

#     def save_image(self):
#         if self.current_frame is not None:
#             filename = f"capture_{int(time.time())}.jpg"
#             cv2.imwrite(filename, self.current_frame)
#             print(f"Image saved: {filename}")

#     def closeEvent(self, event):
#         self.cap.release()

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = CameraApp()
#     window.show()
#     sys.exit(app.exec_())



import sys
import cv2
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Realtime Camera with Trigger")
        self.setGeometry(100, 100, 800, 600)

        # Camera
        self.cap = None
        self.current_frame = None
        self.camera_running = False  # trạng thái camera

        # Giao diện
        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)

        self.btn_camera = QPushButton("Start Camera", self)
        self.btn_camera.clicked.connect(self.toggle_camera)

        self.btn_trigger = QPushButton("Trig-Soft", self)
        self.btn_trigger.clicked.connect(self.save_image)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn_camera)
        layout.addWidget(self.btn_trigger)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Timer update camera
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def toggle_camera(self):
        if not self.camera_running:
            # Mở camera
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
            self.camera_running = True
            self.btn_camera.setText("Stop Camera")
        else:
            # Tắt camera
            self.timer.stop()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.current_frame = None
            self.camera_running = False
            self.btn_camera.setText("Start Camera")
            self.label.clear()

    def update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            # Lật ngang
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qt_image))

    def save_image(self):
        if self.current_frame is not None:
            filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, self.current_frame)
            print(f"Image saved: {filename}")
        else:
            # Camera chưa bật → lưu ảnh trắng
            white_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
            filename = f"capture_white_{int(time.time())}.jpg"
            cv2.imwrite(filename, white_img)
            print(f"No camera running, saved white image: {filename}")

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
