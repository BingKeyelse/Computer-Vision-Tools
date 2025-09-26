import sys
from abc import ABC, abstractmethod
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QMessageBox
)

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


class BaslerCamera(Camera):
    def connect(self):
        self.connected = True
        print(f"Basler {self.name} connected")

    def grab_frame(self):
        if not self.connected:
            raise RuntimeError("Camera not connected")
        return f"Frame from Basler {self.name}"


class USBCamera(Camera):
    def connect(self):
        self.connected = True
        print(f"USB {self.name} connected")

    def grab_frame(self):
        if not self.connected:
            raise RuntimeError("Camera not connected")
        return f"Frame from USB {self.name}"


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
        return f"Blurred({img})"


class CropStep(Step):
    def run(self, img):
        return f"Cropped({img})"


class JudgeStep(Step):
    def run(self, img):
        return f"Judged({img})"


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
        self.manager.add_camera(USBCamera("USB-1"))
        self.manager.add_camera(USBCamera("USB-2"))

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

        QMessageBox.information(self, "Result", f"Pipeline result:\n{result}")


# ------------------ Run App ------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
