from libs import*

# ==== Tool cụ thể: vẽ Circle ====
class CircleTool(MouseTool):# Tool hay diễn viễn Circle
    '''
    Tool để sử dụng cho mục đích vẽ hình tròn
    '''
    def __init__(self):
        """Khởi tạo tool vẽ hình tròn."""

        self.start = None
        self.end = None

        ## Thêm chức năng điều chỉnh
        self.mode = None  # "move", "resize", hoặc None
        self.offset = (0, 0)
        self.start_img= None
        self.end_img= None

    def reset_image(self)-> None:
        """Khởi tạo lại giá trị bắt đầu để reset hình đang vẽ"""
        self.start = None
        self.end = None

    def on_mouse_down(self, event)-> None:
        """Xử lý sự kiện nhấn chuột."""

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

    def len_2_point(self, A: list, B: list)-> float:
        """Tính khoảng cách 2 điểm"""

        len_x= (A[0]-B[0])**2
        len_y= (A[1]-B[1])**2
        return np.sqrt(len_x + len_y)

    def on_mouse_move(self, event, x_offset=0, y_offset=0)-> None:
        """Xử lý sự kiện rê chuột."""

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
        """Xử lý sự kiện nhả chuột."""

        self.mode = None

    def draw(self, painter, x_offset=0, y_offset=0, ratio_base_image=0) -> None:
        """Vẽ hình tròn hiện tại lên canvas, gồm viền, tâm và lớp phủ mờ."""

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

    ## Thêm
    def get_shape(self)-> None:
        """Trả về dữ liệu hình tròn (hoặc None nếu chưa có)."""

        if self.start_img and self.end_img:
            return ("circle", self.start_img, self.end_img)
        return None