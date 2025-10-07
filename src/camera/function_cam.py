from libs import*

class CameraFunctions:
    def __init__(self, ui, cameras, timer, canvas):
        """
        combo   : QComboBox để chọn camera
        label   : QLabel để hiển thị hình ảnh
        timer   : QTimer để update frame liên tục
        cameras : dict chứa danh sách camera {"name": camera_obj}
        """
        self.ui = ui
        self.canvas = canvas
        self.timer = timer
        self.cameras = cameras

        self.active_cam = None
        self.active_name = None

        self.ui.btn_trigsoft.clicked.connect(self.capture_frame)

        # Gắn signal thay đổi camera
        self.ui.btn_choose_cam.addItem("None")
        self.ui.btn_choose_cam.currentTextChanged.connect(self.select_camera)
        self.timer.timeout.connect(self.update_frame)

    def check_cameras(self):
        # 🚫 tạm chặn signal để tránh gọi select_camera('') khi clear
        print("aa")
        self.ui.btn_choose_cam.blockSignals(True)
        self.ui.btn_choose_cam.clear()
        self.ui.btn_choose_cam.addItem("None")

        available = []
        for name, cam in self.cameras.items():
            cam.connect()
            if cam.connected:
                self.ui.btn_choose_cam.addItem(name)
                available.append(name)

        # Nếu camera đang active vẫn còn trong danh sách thì giữ nguyên
        if self.active_name and self.active_name in available:
            self.ui.btn_choose_cam.setCurrentText(self.active_name)
        else:
            # Nếu active không còn thì reset
            self.active_cam = None
            self.active_name = None
            self.ui.btn_choose_cam.setCurrentText("None")
            # self.label.setText("No Camera")
            self.timer.stop()

        self.ui.btn_choose_cam.blockSignals(False)

    def select_camera(self, name):
        if not name or name == "None":  # chống chuỗi rỗng
            self.active_cam = None
            self.active_name = None
            self.canvas.clear_image()
            self.timer.stop()
            return

        if name not in self.cameras:
            return

        cam = self.cameras[name]
        if not cam.connected:
            QMessageBox.warning(self, "Warning", f"{name} chưa được kết nối! Hãy bấm 'Check All Cameras' trước.")
            self.combo.setCurrentText("None")
            return

        self.active_cam = cam
        self.active_name = name
        self.timer.start(30)

    def update_frame(self):
        if self.active_cam:
            frame = self.active_cam.get_frame()
            if frame is not None:
                self.canvas.set_image(frame, link_image = None)
            else:
                # camera mất kết nối khi đang stream
                # self.label.setText("No Camera")
                self.timer.stop()
                self.active_cam = None
                self.active_name = None
                self.ui.btn_choose_cam.setCurrentText("None")

    def capture_frame(self):
        if self.active_cam:
            frame = self.active_cam.get_frame()
            if frame is not None:
                cv2.imshow(f"Captured from {self.active_name}", frame)
                cv2.waitKey(1)

        #     else:
        #         QMessageBox.warning(self, "Warning", "Không lấy được frame từ camera!")
        # else:
        #     QMessageBox.information(self, "Info", "Không có camera nào đang stream.")
