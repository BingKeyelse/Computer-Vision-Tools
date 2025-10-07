import cv2
from abc import ABC, abstractmethod
from pypylon import pylon

class BaseCamera(ABC):
    def __init__(self, name):
        self.name = name
        self.cap = None
        self.connected = False

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def get_frame(self):
        pass

    def disconnect(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False


class USBCamera(BaseCamera):
    def __init__(self, name, index):
        super().__init__(name)
        self.index = index

    def connect(self):
        # Nếu đang kết nối, kiểm tra lại bằng cách đọc thử 1 frame
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

    def get_frame(self):
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

    def disconnect(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False


class BaslerCamera(BaseCamera):
    def __init__(self, name, serial=None):
        super().__init__(name)
        self.serial = serial
        self.camera = None
        self.converter = None

    def connect(self):
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
                        break
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

    def get_frame(self):
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

    def disconnect(self):
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

class CameraFactory:
    registry = {}

    @classmethod
    def register(cls, cam_type, cam_class):
        cls.registry[cam_type] = cam_class

    @classmethod
    def create(cls, cam_type, name, **kwargs):
        return cls.registry[cam_type](name, **kwargs)


CameraFactory.register("usb", USBCamera)
CameraFactory.register("basler", BaslerCamera)


class CreateNameCamera:
    def __init__(self):
        # Khởi tạo tất cả camera ở đây
        self.cameras = {
            "USB Cam 0": CameraFactory.create("usb", "USB Cam 0", index=0),
            "USB Cam 1": CameraFactory.create("usb", "USB Cam 1", index=2),
            "Basler Cam A": CameraFactory.create("basler", "Basler Cam A", serial="21573780"),
            "Basler Cam B": CameraFactory.create("basler", "Basler Cam B", serial="22015482"),
        }