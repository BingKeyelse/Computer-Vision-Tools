import sys
import cv2
import numpy as np
from abc import ABC, abstractmethod
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap

# ------------------ Core OOP ------------------
class Camera(ABC):
    def __init__(self, name):
        self.name = name
        self.connected = False

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def grab_frame(self):
        pass

    def disconnect(self):
        self.connected = False
        print(f"{self.name} disconnected")


class USBCamera(Camera):
    def __init__(self, name, index=0):
        super().__init__(name)
        self.index = index
        self.cap = None

    def connect(self):
        self.cap = cv2.VideoCapture(self.index)
        if self.cap.isOpened():
            self.connected = True
            print(f"USB {self.name} connected")
        else:
            self.connected = False
            print(f"USB {self.name} failed to connect")

    def grab_frame(self):
        if not self.connected or self.cap is None:
            raise RuntimeError("Camera not connected")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to grab frame")
        return frame

    def disconnect(self):
        if self.cap:
            self.cap.release()
        super().disconnect()


# Dummy Basler (chưa có SDK thật, giả lập thôi)
class BaslerCamera(Camera):
    def connect(self):
        self.connected = True
        print(f"Basler {self.name} connected")

    def grab_frame(self):
        if not self.connected:
            raise RuntimeError("Camera not connected")
        # tạo ảnh giả (màu xám)
        return np.zeros((480, 640, 3), dtype=np.uint8)


class CameraManager:
    def __init__(self):
        self.cameras = []

    def add_camera(self, camera: Camera):
        self.cameras.append(camera)

    def check_camera(self, name):
        for cam in self.cameras:
            if cam.name == name:
                cam.connect()
                return cam
        return None


# ---- Pipeline Steps ----
class Step(ABC):
    @abstractmethod
    def run(self, img):
        pass


class BlurStep(Step):
    def run(self, img):
        return cv2.GaussianBlur(img, (3, 3), 0)


class CropStep(Step):
    def run(self, img):
        h, w = img.shape[:2]
        return img[h//4: 3*h//4, w//4: 3*w//4]


class JudgeStep(Step):
    def run(self, img):
        # giả lập phán định: nếu trung bình sáng > 100 thì OK
        mean_val = img.mean()
        text = "OK" if mean_val > 100 else "NG"
        img = img.copy()
        cv2.putText(img, text, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img


class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def run(self, img):
        for step in self.steps:
            img = step.run(img)
        return img


# ------------------ GUI ------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Camera Vision System")

        # Camera manager
        self.manager = CameraManager()
        self.manager.add_camera(BaslerCamera("Basler-1"))
        self.manager.add_camera(BaslerCamera("Basler-2"))
        self.manager.add_camera(USBCamera("USB-1", 0))  # laptop webcam
        self.manager.add_camera(USBCamera("USB-2", 1))  # thêm usb khác nếu có

        self.active_cameras = []

        # Layout
        layout = QVBoxLayout()

        # Camera buttons
        cam_layout = QHBoxLayout()
        for cam in self.manager.cameras:
            btn = QPushButton(f"Check {cam.name}")
            btn.clicked.connect(lambda _, c=cam.name: self.check_camera(c))
            cam_layout.addWidget(btn)
        layout.addLayout(cam_layout)

        # Workflow checkboxes
        self.chk_blur = QCheckBox("Blur")
        self.chk_crop = QCheckBox("Crop")
        self.chk_judge = QCheckBox("Judge")
        layout.addWidget(QLabel("Select Workflow Steps:"))
        layout.addWidget(self.chk_blur)
        layout.addWidget(self.chk_crop)
        layout.addWidget(self.chk_judge)

        # Run button
        run_btn = QPushButton("Run Auto")
        run_btn.clicked.connect(self.run_auto)
        layout.addWidget(run_btn)

        # Image display
        self.image_label = QLabel("No Image")
        self.image_label.setFixedSize(640, 480)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

    def check_camera(self, name):
        cam = self.manager.check_camera(name)
        if cam and cam.connected:
            self.active_cameras.append(cam)
            QMessageBox.information(self, "Camera", f"{name} Ready ✅")
        else:
            QMessageBox.warning(self, "Camera", f"{name} Failed ❌")

    def run_auto(self):
        if not self.active_cameras:
            QMessageBox.warning(self, "Error", "No camera connected!")
            return

        # Lấy camera đầu tiên đang active
        cam = self.active_cameras[0]
        frame = cam.grab_frame()

        # Build pipeline theo checkbox
        steps = []
        if self.chk_blur.isChecked():
            steps.append(BlurStep())
        if self.chk_crop.isChecked():
            steps.append(CropStep())
        if self.chk_judge.isChecked():
            steps.append(JudgeStep())

        pipeline = Pipeline(steps)
        result = pipeline.run(frame)

        # Convert OpenCV image → QPixmap để hiển thị
        self.display_image(result)

    def display_image(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix.scaled(
            self.image_label.width(), self.image_label.height()
        ))


# ------------------ Run App ------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
