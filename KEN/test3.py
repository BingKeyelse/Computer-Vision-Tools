import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt

class ImageViewer(QLabel):
    def __init__(self, img):
        super().__init__()
        self.img = img                     # ảnh gốc (BGR)
        self.img_h, self.img_w = img.shape[:2]
        self.box_start = None              # điểm bắt đầu
        self.box_end = None                # điểm kết thúc
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: black;")  # nền đen

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # click đầu tiên -> bắt đầu box
            if self.box_start is None:
                gx, gy = self.map_to_original(event.pos())
                if gx is not None:
                    self.box_start = (gx, gy)
                    self.box_end = None
            else:
                # click lần 2 -> kết thúc box
                gx, gy = self.map_to_original(event.pos())
                if gx is not None:
                    self.box_end = (gx, gy)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.img is None:
            return

        # Tính scale và offset để hiển thị ảnh
        label_w, label_h = self.width(), self.height()
        scale = min(label_w / self.img_w, label_h / self.img_h)
        scaled_w, scaled_h = int(self.img_w * scale), int(self.img_h * scale)
        offset_x, offset_y = (label_w - scaled_w) // 2, (label_h - scaled_h) // 2

        # Convert OpenCV BGR -> Qt RGB
        rgb_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb_img.data, self.img_w, self.img_h, self.img_w*3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Vẽ ảnh
        painter = QPainter(self)
        painter.drawPixmap(offset_x, offset_y, pixmap)

        # Nếu có box thì vẽ
        if self.box_start and self.box_end:
            # Map box từ gốc sang màn hình
            x1, y1 = self.box_start
            x2, y2 = self.box_end
            sx1 = int(x1 * scale + offset_x)
            sy1 = int(y1 * scale + offset_y)
            sx2 = int(x2 * scale + offset_x)
            sy2 = int(y2 * scale + offset_y)

            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawRect(min(sx1, sx2), min(sy1, sy2),
                             abs(sx2 - sx1), abs(sy2 - sy1))

        painter.end()

    def map_to_original(self, pos):
        """Chuyển từ tọa độ chuột (QLabel) -> tọa độ ảnh gốc"""
        label_w, label_h = self.width(), self.height()
        scale = min(label_w / self.img_w, label_h / self.img_h)
        scaled_w, scaled_h = int(self.img_w * scale), int(self.img_h * scale)
        offset_x, offset_y = (label_w - scaled_w) // 2, (label_h - scaled_h) // 2

        mx, my = pos.x() - offset_x, pos.y() - offset_y
        if 0 <= mx < scaled_w and 0 <= my < scaled_h:
            gx = int(mx / scale)
            gy = int(my / scale)
            return gx, gy
        return None, None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Annotation Example")
        self.showMaximized()  # full màn hình

        # Load ảnh test
        img = np.full((480, 640, 3), 200, dtype=np.uint8)  # ảnh xám
        cv2.putText(img, "Test Image", (100, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        self.viewer = ImageViewer(img)

        layout = QVBoxLayout()
        layout.addWidget(self.viewer)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
