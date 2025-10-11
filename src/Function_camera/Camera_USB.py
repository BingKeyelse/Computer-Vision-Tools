from libs import*

class USBCamera(BaseCamera):
    def __init__(self, name, index):
        """
        ## Khởi tạo USB với đầu vào
        - input:
            - name: Tên truyền vào
            - index: chỉ định index để phân loại khác biệt ra
        """
        super().__init__(name) # Gọi tới phần khởi tạo init của BaseCamera luôn, được tái sử dụng và thừa kế trong đây luôn
        '''
        self.name = name
        self.cap = None
        self.connected = False
        '''
        self.index = index

    def connect(self, *args, **kwargs):
        """## Kiểm tra kết nối
        - Nếu đang kết nối, kiểm tra lại bằng cách đọc thử 1 frame\
        """
        if self.cap and self.cap.isOpened():
            ret, _ = self.cap.read()
            if ret:
                self.connected = True
                return
            else:
                # Không đọc được frame -> mất kết nối
                self.cap.release()
                self.cap = None
                self.connected = False

        # Nếu cap tồn tại nhưng không mở được thì giải phóng
        if self.cap:
            self.cap.release()
            self.cap = None

        # Thử mở lại
        self.cap = cv2.VideoCapture(self.index)
        if self.cap.isOpened():
            ret, _ = self.cap.read()
            if ret:
                self.connected = True
                return
        # Nếu tới đây thì fail
        self.connected = False
        if self.cap:
            self.cap.release()
        self.cap = None

    def get_frame(self) -> np.ndarray | None:
        """## Lấy frame ảnh"""
        if self.connected and self.cap:
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                # mất kết nối giữa chừng
                self.connected = False
                if self.cap:
                    self.cap.release()
                    self.cap = None
        return None

    # def disconnect(self):# Check lại phần này xem bỏ được không nhé vì thừa kế mà 
    #     """## Hủy kết nối"""
    #     if self.cap:
    #         self.cap.release()
    #         self.cap = None
    #     self.connected = False