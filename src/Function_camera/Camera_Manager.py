from libs import*

class CameraFactory:
    """
    ## Nhà máy camera để đăng kí và gán thuộc tính hay phân loại kiểu của chúng vào với từng camera được chỉ định
    """
    registry = {} #-> Đây tồn tại như một bảng tra cứu

    @classmethod # Gọi trực tiếp qua class mà không cần kiểu là object như self. Giúp cho không cần khởi tạo ở hàm bên dưới
    def register(cls, cam_type, cam_class):
        """
        ## Gắn class với key cho class phù hợp 'USBCamera' | 'BaslerCamera'
        - input
            - class: 'usb' hay 'basler' khai báo Key
            - type camera: gán tên phù hợp
            - cam_class: gán cho Class của từng cam với Key tương ứng phía trên
      
        """
        cls.registry[cam_type] = cam_class

    @classmethod
    def create(cls, cam_type, name, **kwargs)-> Union['USBCamera', 'BaslerCamera']:
        """
        ## Truyền đối số vào class với tùy chọn là key 
        - input
            - cls: 'usb' hay 'basler' lấy Key
            - type camera: gán tên phù hợp
            - name: dùng để phân biệt 
        - output
            - Là đối tượng USBCamera hoặc BaslerCamera

        """
        return cls.registry[cam_type](name, **kwargs)


class CreateNameCamera:
    def __init__(self):
        """
        ## Khởi tạo Camera USB and Basler gồm 
        - class: 'usb' hay 'basler'
        - type camera: gán tên phù hợp
        - index or serial: dùng để phân biệt bằng index hay serial
        """
        # Gán hàm với Key đã
        CameraFactory.register("usb", USBCamera)
        CameraFactory.register("basler", BaslerCamera)

        # Với Key tùy biến truyền paremeter tùy chọn
        self.cameras = {
            "USB Cam 0": CameraFactory.create("usb", "USB Cam 0", index=0),
            "USB Cam 1": CameraFactory.create("usb", "USB Cam 1", index=2),
            "Basler Cam A": CameraFactory.create("basler", "Basler Cam A", serial="21573780"),
            "Basler Cam B": CameraFactory.create("basler", "Basler Cam B", serial="22015482"),
        }