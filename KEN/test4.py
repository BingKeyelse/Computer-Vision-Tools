import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import QTimer, Qt, QRect

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Tool with Box Annotation")
        self.showMaximized()  # Full màn hình

        # QLabel hiển thị ảnh
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        # Nút nhấn
        self.btn_realtime = QPushButton("Realtime")
        self.btn_trigger = QPushButton("Trig-Soft")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn_realtime)
        layout.addWidget(self.btn_trigger)
        self.setLayout(layout)

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Biến box
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.boxes = []  # lưu box theo ảnh gốc
        self.current_frame = None

        # Sự kiện nút
        self.btn_realtime.clicked.connect(self.toggle_realtime)
        self.btn_trigger.clicked.connect(self.trigger_soft)

    def toggle_realtime(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(30)

    def trigger_soft(self):
        if self.current_frame is not None:
            frame = self.current_frame.copy()

            # Vẽ box lên frame gốc khi lưu
            for box in self.boxes:
                cv2.rectangle(frame, box[0], box[1], (0, 0, 255), 2)

            cv2.imwrite("capture.jpg", frame)
            print("Saved capture.jpg with boxes.")
        else:
            blank = np.ones((480, 640, 3), dtype=np.uint8) * 255
            cv2.imwrite("blank.jpg", blank)
            print("Saved blank.jpg (no camera).")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.display_image(frame)

    def display_image(self, img):
        """Hiển thị ảnh lên QLabel, scale theo kích thước label"""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg)
        self.label.setPixmap(pix.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.current_frame is not None:
            self.drawing = True
            self.start_point = self.map_to_original(event.pos())

    def mouseMoveEvent(self, event):
        if self.drawing and self.current_frame is not None:
            self.end_point = self.map_to_original(event.pos())
            self.repaint()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.end_point = self.map_to_original(event.pos())
            if self.start_point and self.end_point:
                self.boxes.append((self.start_point, self.end_point))
            self.repaint()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.current_frame is not None and (self.start_point and self.end_point):
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))

            # Scale lại box theo màn hình để vẽ overlay
            img_h, img_w, _ = self.current_frame.shape
            label_w = self.label.width()
            label_h = self.label.height()

            scale_x = label_w / img_w
            scale_y = label_h / img_h

            for box in self.boxes + ([(self.start_point, self.end_point)] if self.drawing else []):
                (x1, y1), (x2, y2) = box
                rect = QRect(int(x1 * scale_x), int(y1 * scale_y), 
                             int((x2 - x1) * scale_x), int((y2 - y1) * scale_y))
                painter.drawRect(rect)

    def map_to_original(self, pos):
        """Map tọa độ chuột về ảnh gốc"""
        if self.current_frame is None:
            return None
        img_h, img_w, _ = self.current_frame.shape
        label_w = self.label.width()
        label_h = self.label.height()

        x_ratio = img_w / label_w
        y_ratio = img_h / label_h

        return (int(pos.x() * x_ratio), int(pos.y() * y_ratio))

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CameraApp()
    win.show()
    sys.exit(app.exec_())
