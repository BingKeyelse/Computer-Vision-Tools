from libs import*

# ==== Tool cụ thể: vẽ Circle ====
class CircleTool(MouseTool):# Tool hay diễn viễn Circle
    '''
    ## Tool để sử dụng cho mục đích vẽ hình tròn
    '''
    def __init__(self):
        """
        Khởi tạo giá trị bắt đầu start và end
        """
        self.start = None
        self.end = None

        ## Thêm chức năng điều chỉnh
        self.mode = None  # "move", "resize", hoặc None
        self.offset = (0, 0)
        self.start_img= None
        self.end_img= None

        self.start_img_100= None
        self.end_img_100= None

        self.scale_resize= 1.0

    def reset_image(self)-> None:
        """## Khởi tạo lại giá trị bắt đầu để reset hình đang vẽ"""
        self.start = None
        self.end = None

    def on_mouse_down(self, event)-> None:
        """
        ## Thời điểm khi ấn xuống là xem có gần tâm của hình, hay cạnh nào không
        - Nếu không khai báo giá trị toạ độ cho cả start và end mới
        - start: x,y
        - end:   x,y
        """

        if not self.start or not self.end:
            # nếu chưa có hình thì khởi tạo
            self.start = (event.x(), event.y())
            self.end = self.start
            return
        
        ## Thêm chức năng điều chỉnh
        x, y = event.x(), event.y()
        cx, cy = self.start
        r = self.len_2_point(self.start, self.end)
        d = self.len_2_point((x, y), (cx, cy))

        if abs(d) < 10:   # click gần tâm
            self.mode = "move"
            self.offset = (x - cx, y - cy)
        elif abs(d - r) < 10:  # click gần biên
            self.mode = "resize"
        else:
            # coi như bắt đầu vẽ mới (nếu muốn cho phép vẽ lại)
            self.start = (x, y)
            self.end = self.start
            self.mode = None

    def on_mouse_move(self, event)-> None:
        """
        ## Khi chuột di chuyển thì cập nhập lại với từ mode đang ở hiện tại
        - Nếu không ở mode nào thì cập nhập lại end
        - end: x,y cập nhập theo chuột

        """

        if not self.start or not self.end:
            return
        
        ## Thêm chức năng điều chỉnh 
        x, y = event.x(), event.y()
        if self.mode == "move":
            x1, y1 = self.start
            x2, y2 = self.end
            dx, dy = x - self.offset[0] - x1, y - self.offset[1] - y1
            self.start = (x1 + dx, y1 + dy)
            self.end = (x2 + dx, y2 + dy)

        elif self.mode == "resize":
            # end = điểm mới trên biên → bán kính thay đổi
            self.end = (x, y)

        else:
            # vẽ lần đầu
            self.end = (x, y)

    def on_mouse_up(self, event)-> None:
        """
        ## Nếu thả ra thì đưa mode về cơ bản
        
        """

        self.mode = None

    def draw(self, painter, x_offset=0, y_offset=0, ratio_base_image=0, scale=1.0) -> None:
        """
        ## Vẽ hình tròn với khung đỏ, tâm hình và nền mờ trong suốt.
        - Hình tròn được vẽ bởi painter lấy từ ToolManager truyền cho
        
        """

        # Lấy scale resize kích thước ảnh
        self.scale_resize= scale

        if self.start and self.end:
            x1, y1 = self.start
            x2, y2 = self.end

            # Trừ offset để ra tọa độ trên ảnh scaled
            self.start_img_scaled = (x1 - x_offset, y1 - y_offset)
            self.end_img_scaled   = (x2 - x_offset, y2 - y_offset)

            # Tọa độ trên ảnh thực tế
            # Đây là bước tính toán để truyền thẳng cho self.get_shape vì tính global của biến
            self.start_img = (self.start_img_scaled[0] / ratio_base_image[0],
                  self.start_img_scaled[1] / ratio_base_image[1])
            self.end_img = (self.end_img_scaled[0] / ratio_base_image[0],
                self.end_img_scaled[1] / ratio_base_image[1])
            
            # Tọa độ trên ảnh size 100%
            self.start_img_100 = (self.start_img_scaled[0] / (ratio_base_image[0]*self.scale_resize),
                  self.start_img_scaled[1] / (ratio_base_image[1]*self.scale_resize))
            self.end_img_100 = (self.end_img_scaled[0] / (ratio_base_image[0]*self.scale_resize),
                self.end_img_scaled[1] / (ratio_base_image[1]*self.scale_resize))

            # Tính bán kính = khoảng cách từ start đến end
            r = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)

            # Vẽ viền đỏ
            pen = QPen(Qt.red, 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(x1 - r, y1 - r, 2 * r, 2 * r)

            # Vẽ tâm màu vàng
            pen = QPen(Qt.yellow, 2)
            painter.setPen(pen)
            painter.setBrush(QBrush(Qt.yellow))
            painter.drawEllipse(x1 - 3, y1 - 3, 6, 6)

            # Lớp phủ đỏ trong suốt
            brush = QBrush(QColor(255, 0, 0, 50))  # đỏ, alpha=50
            painter.setBrush(brush)
            painter.setPen(Qt.NoPen)  # bỏ viền khi fill
            painter.drawEllipse(x1 - r, y1 - r, 2 * r, 2 * r)

            # Reset lại brush/pen để không ảnh hưởng cái vẽ sau
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(Qt.red, 2))

    def get_shape(self)-> tuple[str, tuple[int,int], tuple[int,int]] | None:
        """
        ## Lấy tên của shape và giá trị start, end tuyệt đối trên ảnh kích thước thực tế để lưu và đặc biệt là với tọa độ ảnh 100%
        - output : (shape: str, start_100, end_100)
        """
        if self.start_img_100 and self.end_img_100:
            return ("circle", self.start_img_100, self.end_img_100)
        return None
    
    def len_2_point(self, A: list, B: list)-> float:
        """## Tính khoảng cách 2 điểm
        - input
            - A, B: tọa độ 2 điểm cần tính
        - output
            - Khoảng cách 2 điểm
        """

        len_x= (A[0]-B[0])**2
        len_y= (A[1]-B[1])**2
        return np.sqrt(len_x + len_y)