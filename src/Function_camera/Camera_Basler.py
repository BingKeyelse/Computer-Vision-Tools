from libs import*

class BaslerCamera(BaseCamera):
    def __init__(self, name, serial=None):
        """
        ## Khởi tạo Basler với đầu vào
        - input:
            - name: Tên truyền vào
            - serial: dãy định danh của camera Basler
        """
        super().__init__(name)
        self.serial = serial
        self.camera = None
        self.converter = None

    def connect(self):
        """
        ## Kiểm tra thiết bị kết nối camera Basler
        - Kiểm tra só sánh với serial và xong sau đó thì gán self.camera và self.converter
        """
        print('kêt nối được luôn')
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            if not devices:
                self.connected = False
                return

            # Tìm đúng camera theo serial
            device = None
            if self.serial:
                for d in devices:
                    if d.GetSerialNumber() == self.serial:
                        device = d
                        return
            else:
                device = devices[0]  # lấy camera đầu tiên

            if device is None:
                self.connected = False
                return

            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(device))
            self.camera.Open()
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            # Cấu hình converter để ra BGR
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            self.connected = True

        except Exception as e:
            print(f"Basler connect error: {e}")
            self.connected = False
            if self.camera:
                self.camera.Close()
            self.camera = None

    def get_frame(self)-> np.ndarray| None:
        """
        ## Lấy frame của Camera Basler với timeout 1s
        - output: frame của khung hình
        """
        if self.connected and self.camera and self.camera.IsGrabbing():
            grab = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            if grab.GrabSucceeded():
                img = self.converter.Convert(grab)  # chuyển định dạng
                frame = img.GetArray()              # numpy array (BGR)
                grab.Release()
                return frame
            else:
                grab.Release()
                self.connected = False
        return None

    def disconnect(self) -> None:
        """
        ## Hàm này để ngừng Camera
        """
        if self.camera:
            try:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                if self.camera.IsOpen():
                    self.camera.Close()
            except Exception:
                pass
        self.camera = None
        self.connected = False