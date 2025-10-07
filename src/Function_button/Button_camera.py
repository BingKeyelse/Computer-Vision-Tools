from libs import*

class CameraFunctions:
    def __init__(self, ui, cameras, timer_0, canvas):
        """
        ## Kế thừa các giá trị từ mainWindow
        - ui: phần giao diện thừa kế
        - cameras: thừa kế toàn bộ giá trị Camera
        - timer: bộ đếm timer
        - canvas: phần thừa kế để hiện thị ảnh ở phần chính giữa để chạy phần camera Realtime
        
        """
        self.ui = ui
        self.canvas = canvas
        self.timer_0 = timer_0
        self.cameras = cameras

        self.active_cam = None
        self.active_name = None

        self.ui.btn_check_cam.clicked.connect(self.check_cameras)

        self.ui.btn_trigsoft.clicked.connect(self.capture_frame)

        # Gắn signal thay đổi camera
        self.ui.btn_choose_cam.addItem("None")
        self.ui.btn_choose_cam.currentTextChanged.connect(self.select_camera)
        self.timer_0.timeout.connect(self.update_frame)

    def check_cameras(self)-> None:
        """
        ## Kiểm tra camera check xem cái nào đang có
        """
        # 🚫 Tạm chặn signal để tránh gọi select_camera('') khi clear
        self.ui.btn_choose_cam.blockSignals(True)
        self.ui.btn_choose_cam.clear()
        self.ui.btn_choose_cam.addItem("None")

        available = []
        for name, cam in self.cameras.items():
            cam.connect() # Cho chúng nó kết nối hết luôn đi
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
            self.timer_0.stop()

        self.ui.btn_choose_cam.blockSignals(False)

    def select_camera(self, name) -> None:
        """
        ## Lựa chọn camera sẽ đưuọc hiển thị 
        - name: tên của đối tượng camera sẽ được tự động tuyền vào
        """
        if not name or name == "None":  
            '''
            Khi mà không có cam thì clear nó đi và nhớ cập nhập ở canvas
            '''
            self.active_cam = None
            self.active_name = None
            self.canvas.clear_image()
            self.timer_0.stop()
            return

        if name not in self.cameras:
            return

        cam = self.cameras[name] # Lấy toàn bộ đối tượng ra luôn
        if not cam.connected:
            QMessageBox.warning(self, "Warning", f"{name} chưa được kết nối! Hãy bấm 'Check All Cameras' trước.")
            self.combo.setCurrentText("None")
            return

        self.active_cam = cam
        self.active_name = name
        self.timer_0.start(30)

    def update_frame(self):
        """
        ## Hiển thị video thu thập lên trên canvas + Timer 0
        """
        if self.active_cam:
            frame = self.active_cam.get_frame()
            if frame is not None: 
                self.canvas.set_image(frame, link_image = None)
            else:
                '''Camera mất kết nối khi đang stream'''
                self.timer_0.stop()
                self.active_cam = None
                self.active_name = None
                self.ui.btn_choose_cam.setCurrentText("None")

    def capture_frame(self)-> None:
        """
        Dùng để chụp ảnh lại và sẽ lưu vào folder
        """
        if self.active_cam:
            frame = self.active_cam.get_frame()
            if frame is not None:
                cv2.imshow(f"Captured from {self.active_name}", frame)
                cv2.waitKey(1)

