from libs import*

class CameraChecker(QThread):
    finished = pyqtSignal(list)

    def __init__(self, cameras, devices, tl_factory, ui):
        """
        ## Scan camera đang có
        - Args:
            - cameras: truyền vào thông tin camera được khai báo để quét
        """
        super().__init__()
        self.cameras = cameras
        self.devices = devices
        self.tl_factory = tl_factory
        self.ui = ui

    def run(self):
        """
        ## Emit data những cam được quét để trở về hàm kết nối
        """
        self.ui.btn_choose_cam.blockSignals(True)
        self.ui.btn_choose_cam.clear()
        self.ui.btn_choose_cam.addItem("None")

        available = []
        for name, cam in self.cameras.items():                         
            cam.connect(self.devices, self.tl_factory)
            if cam.connected:
                self.ui.btn_choose_cam.addItem(name)
                available.append(name)
        self.finished.emit(available)  # gửi lại kết quả

class CameraFunctions:
    def __init__(self, ui, cameras, timer_0, canvas_Camera):
        """
        ## Kế thừa các giá trị từ mainWindow
        - ui: phần giao diện thừa kế
        - cameras: thừa kế toàn bộ giá trị Camera
        - timer: bộ đếm timer
        - canvas_Camera: phần thừa kế để hiện thị ảnh ở phần chính giữa để chạy phần camera Realtime
        """

        self.ui = ui
        self.canvas_Camera = canvas_Camera
        self.timer_0 = timer_0
        self.cameras = cameras

        self.active_cam = None
        self.active_name = None

        self.ui.btn_check_cam.clicked.connect(self.init_cam)

        self.ui.btn_trigsoft.clicked.connect(self.capture_frame)

        # Gắn signal thay đổi camera
        self.ui.btn_choose_cam.addItem("None")
        self.ui.btn_choose_cam.currentTextChanged.connect(self.select_camera)
        self.timer_0.timeout.connect(self.update_frame)
    
    def init_cam(self):
        """
        - Kiem tra so luong camera o trong thread main de tranh loi
        - Sau do goi phuong thuc check_cameras()
        """
        self.ui.btn_check_cam.setText("🔄Checking...")
        self.timer_0.stop()
        self.tl_factory = pylon.TlFactory.GetInstance()
        self.devices = self.tl_factory.EnumerateDevices()
        if not self.devices:
            self.connected = False
        self.check_cameras()
    
    def check_cameras(self):
        """
        ## Kiểm tra xem có camera nào đang được mở không
        - Kết nới với một thread để check camera
        - Emit singal đến hàm self._on_check_done để cập nhập giao diện
        """
        

        self.thread = CameraChecker(self.cameras, self.devices, self.tl_factory, self.ui)
        self.thread.finished.connect(self._on_check_done)
        self.thread.start()  # ✅ start mà không block UI
    
    def _on_check_done(self, available):
        """
        ## Sau khi check cam xong thì điều chỉnh giao diện
        - Đoạn này có check thông tin xem có đang mở cái cam được chọn không đó ấy nhé
        - Inputs:
            - available: là thông tin cam nhận được từ Thread Scan Camera - singal
        """
        # ✅ Tương tự như sau khi join — bạn xử lý ở đây
        # self.ui.btn_choose_cam.blockSignals(True)
        # self.ui.btn_choose_cam.clear()
        # self.ui.btn_choose_cam.addItem("None")
        self.ui.btn_check_cam.setText("Check Cam")

        # for name in available:
        #     self.ui.btn_choose_cam.addItem(name)
        
        if self.active_name and self.active_name in available:
            self.ui.btn_choose_cam.setCurrentText(self.active_name)
        else:
            self.active_cam = None
            self.active_name = None
            self.ui.btn_choose_cam.setCurrentText("None")
            self.timer_0.stop()

        
        self.ui.stackwidget.setCurrentWidget(self.ui.page_main)
        self.ui.btn_choose_cam.blockSignals(False)
        self.thread.quit()
        self.thread.wait()  # tương đương join nhưng không block UI
        self.timer_0.start(30)

    def select_camera(self, name) -> None:
        """
        ## Lựa chọn camera sẽ đưuọc hiển thị 
        - name: tên của đối tượng camera sẽ được tự động tuyền vào
        """
        if not name or name == "None":  
            '''
            Khi mà không có cam thì clear nó đi và nhớ cập nhập ở canvas_Camera
            '''
            self.active_cam = None
            self.active_name = None
            self.canvas_Camera.clear_image()
            self.timer_0.stop()
            return

        if name not in self.cameras:
            return
        
        cam = self.cameras[name] # Lấy toàn bộ đối tượng ra luôn
        if not cam.connected:
            # QMessageBox.warning(self, "Warning", f"{name} chưa được kết nối! Hãy bấm 'Check All Cameras' trước.")
            # self.combo.setCurrentText("None")
            return None

        self.active_cam = cam
        self.active_name = name
        self.timer_0.start(30)

    def update_frame(self):
        """
        ## Hiển thị video thu thập lên trên canvas_Camera + Timer 0
        """
        if self.active_cam:
            frame = self.active_cam.get_frame()
            if frame is not None: 
                self.canvas_Camera.set_image(frame, link_image = None)
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

