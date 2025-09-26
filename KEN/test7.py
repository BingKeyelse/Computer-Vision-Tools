import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QLabel, QCheckBox, QGroupBox, QGridLayout, QMessageBox
)

# ========================
# Camera OOP
# ========================
class Camera:
    def __init__(self, name, index=None):
        self.name = name
        self.index = index
        self.cap = None

    def connect(self):
        raise NotImplementedError

    def disconnect(self):
        if self.cap and isinstance(self.cap, cv2.VideoCapture):
            self.cap.release()
        self.cap = None

    def is_connected(self):
        if isinstance(self.cap, cv2.VideoCapture):
            return self.cap.isOpened()
        return self.cap is not None


class USBCamera(Camera):
    def connect(self):
        self.cap = cv2.VideoCapture(self.index)
        return self.is_connected()

    def capture(self):
        if self.is_connected():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None


class BaslerCamera(Camera):
    def connect(self):
        print(f"Connecting to Basler camera {self.name} ... (mock)")
        self.cap = True   # mock connected
        return True

    def capture(self):
        # giả lập tạo ảnh test
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.putText(img, self.name, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return img


# ========================
# Camera Factory
# ========================
class CameraFactory:
    registry = {}

    @classmethod
    def register(cls, cam_type, cam_class):
        cls.registry[cam_type] = cam_class

    @classmethod
    def create(cls, cam_type, name, **kwargs):
        if cam_type not in cls.registry:
            raise ValueError(f"Unknown camera type: {cam_type}")
        return cls.registry[cam_type](name, **kwargs)


CameraFactory.register("usb", USBCamera)
CameraFactory.register("basler", BaslerCamera)


# ========================
# GUI
# ========================
class CameraGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera System with Workflow")

        # Khởi tạo 4 camera
        self.cameras = {
            "USB Cam 0": CameraFactory.create("usb", "USB Cam 0", index=0),
            "USB Cam 1": CameraFactory.create("usb", "USB Cam 1", index=1),
            "Basler Cam A": CameraFactory.create("basler", "Basler Cam A"),
            "Basler Cam B": CameraFactory.create("basler", "Basler Cam B"),
        }

        # Layout chính
        layout = QVBoxLayout()

        # --- Camera Check Section ---
        cam_group = QGroupBox("Check Cameras")
        cam_layout = QGridLayout()
        self.status_labels = {}

        row = 0
        for cam_name, cam in self.cameras.items():
            btn = QPushButton(f"Check {cam_name}")
            lbl = QLabel("Not checked")
            self.status_labels[cam_name] = lbl

            btn.clicked.connect(lambda _, c=cam, n=cam_name: self.check_camera(c, n))
            cam_layout.addWidget(btn, row, 0)
            cam_layout.addWidget(lbl, row, 1)
            row += 1

        cam_group.setLayout(cam_layout)
        layout.addWidget(cam_group)

        # --- Workflow Section ---
        wf_group = QGroupBox("Workflow Steps")
        self.cb_capture = QCheckBox("Capture Image")
        self.cb_blur = QCheckBox("Blur")
        self.cb_crop = QCheckBox("Crop (center 100x100)")
        self.cb_classify = QCheckBox("Classify (OK/NG)")

        for cb in [self.cb_capture, self.cb_blur, self.cb_crop, self.cb_classify]:
            cb.setChecked(True)  # mặc định bật
        wf_layout = QVBoxLayout()
        for cb in [self.cb_capture, self.cb_blur, self.cb_crop, self.cb_classify]:
            wf_layout.addWidget(cb)
        wf_group.setLayout(wf_layout)
        layout.addWidget(wf_group)

        # --- Run Auto Button ---
        self.run_btn = QPushButton("Run Auto")
        self.run_btn.clicked.connect(self.run_auto)
        layout.addWidget(self.run_btn)

        self.setLayout(layout)

    def check_camera(self, cam, cam_name):
        connected = cam.connect()
        if connected:
            self.status_labels[cam_name].setText("✅ Connected")
        else:
            self.status_labels[cam_name].setText("❌ Not Connected")

    def run_auto(self):
        # tìm camera đã kết nối
        active_cams = [c for c in self.cameras.values() if c.is_connected()]
        if not active_cams:
            QMessageBox.warning(self, "Warning", "No connected cameras!")
            return

        for cam in active_cams:
            print(f"\n--- Running workflow for {cam.name} ---")
            img = None

            # Step 1: Capture
            if self.cb_capture.isChecked():
                img = cam.capture()
                if img is None:
                    print(f"{cam.name}: Failed to capture")
                    continue
                print(f"{cam.name}: Image captured")

            # Step 2: Blur
            if self.cb_blur.isChecked() and img is not None:
                img = cv2.GaussianBlur(img, (5, 5), 0)
                print(f"{cam.name}: Blurred")

            # Step 3: Crop
            if self.cb_crop.isChecked() and img is not None:
                h, w = img.shape[:2]
                ch, cw = h // 2, w // 2
                img = img[ch-50:ch+50, cw-50:cw+50]
                print(f"{cam.name}: Cropped")

            # Step 4: Classify
            if self.cb_classify.isChecked() and img is not None:
                result = "OK" if np.mean(img) > 50 else "NG"
                print(f"{cam.name}: Classified as {result}")

            # Hiển thị ảnh cuối cùng
            if img is not None:
                cv2.imshow(cam.name, img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ========================
# Main
# ========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = CameraGUI()
    gui.show()
    sys.exit(app.exec_())
