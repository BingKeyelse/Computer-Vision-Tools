from libs import*

# ==== Tool cụ thể: vẽ Box ====
class BoxTool(MouseTool): # Tool hay diễn viễn Box
    '''
    Tool để sử dụng cho mục đích vẽ hình chữ nhật
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


    def on_mouse_down(self, event, x_offset=0, y_offset=0)-> None:
        """
        Thời điểm khi ấn xuống là xem có gần tâm của hình, hay cạnh nào không
        Nếu không khai báo giá trị toạ độ cho cả start và end mới
        start: x,y
        end:   x,y
        
        """
        # Khởi tạo lúc đầu
        if not self.start or not self.end:
            # nếu chưa có hình thì khởi tạo
            self.start = (event.x(), event.y())
            self.end = self.start
            return
        
        # Lấy toạ độ 
        x, y = event.x(), event.y()
        x1, y1 = self.start
        x2, y2 = self.end

        # Toạ độ tâm chữ nhật
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2) 

        # Khoảng cách đến tâm
        if abs(x - cx) < 10 and abs(y - cy) < 10:
            self.mode = "move"
            self.offset = (x - cx, y - cy) # Khoảng cách cần offset
        # Khoảng cách đến cạnh dọc
        elif min(abs(x - x1), abs(x - x2)) < 10 and (y1 <= y <= y2):
            self.mode = "resize-x"
        # Khoảng cách đến cạnh ngang
        elif min(abs(y - y1), abs(y - y2)) < 10 and (x1 <= x <= x2):
            self.mode = "resize-y"
        # Khi không phải điều chỉnh thì reset tạo mới
        else:
            self.mode = None
            self.start = (event.x(), event.y())
            self.end = self.start


    def on_mouse_move(self, event, x_offset=0, y_offset=0, scaled=0.0)-> None:
        """
        Khi chuột di chuyển thì cập nhập lại với từ mode đang ở hiện tại
        Nếu không ở mode nào thì cập nhập lại end
        end: x,y cập nhập theo chuột

        """
        if not self.start or not self.end:
            return

        x, y = event.x(), event.y()
        # print(f"Giá trị thực sự của điểm end là {(x,y)}")
        # print('Sao không chạy ở đây nữa')
        # print(x - x_offset, y - y_offset) # Trả về lại tọa độ so với ảnh đang nằm bên trong widget
        if self.mode == "move":
            # dịch chuyển cả hình
            x1, y1 = self.start
            x2, y2 = self.end
            w, h = x2 - x1, y2 - y1
            cx, cy = x - self.offset[0], y - self.offset[1] # Toạ độ của hình chứ nhật mới
            self.start = (int(cx - w/2), int(cy - h/2)) # TÍnh toán lại start của hình chữ nhật
            self.end   = (int(cx + w/2), int(cy + h/2)) # Tính toán lại end của hình chữ nhật 

        elif self.mode == "resize-x":
            # thay đổi chiều ngang
            x1, y1 = self.start
            x2, y2 = self.end
            if abs(x - x1) < abs(x - x2):
                self.start = (x, y1)
            else:
                self.end = (x, y2)

        elif self.mode == "resize-y":
            # thay đổi chiều dọc
            x1, y1 = self.start
            x2, y2 = self.end
            if abs(y - y1) < abs(y - y2):
                self.start = (x1, y)
            else:
                self.end = (x2, y)

        else:
            # trường hợp chưa chọn mode → vẽ như bình thường
            self.end = (x, y)


    def on_mouse_up(self, event)-> None:
        """
        Nếu thả ra thì đưa mode về cơ bản
        
        """
        self.mode = None
        

    def draw(self, painter, x_offset=0, y_offset=0, ratio_base_image=0)-> None:
        """
        Vẽ hình chữ nhật với khung đỏ, tâm hình và nền mờ trong suốt.
        Hình chữ nhật được vẽ bởi painter lấy từ ToolManager truyền cho
        
        """
        ox, oy = x_offset, y_offset  # offset khi căn giữa ảnh
        if self.start and self.end:
            x1, y1 = self.start
            x2, y2 = self.end

            print(f'Giá trị offset trên này==== {x_offset} {y_offset}')

            # Trừ offset để ra tọa độ trên ảnh scaled
            self.start_img_scaled = (x1 - x_offset, y1 - y_offset)
            self.end_img_scaled   = (x2 - x_offset, y2 - y_offset)

            # Tọa độ trên ảnh thực tế
            self.start_img = (self.start_img_scaled[0] / ratio_base_image[0],
                  self.start_img_scaled[1] / ratio_base_image[1])
            self.end_img = (self.end_img_scaled[0] / ratio_base_image[0],
                self.end_img_scaled[1] / ratio_base_image[1])

            # Chuẩn hóa tọa độ (tránh trường hợp kéo ngược chuột)
            left, top = min(x1, x2), min(y1, y2)
            width, height = abs(x2 - x1), abs(y2 - y1)

            # 1. Vẽ khung viền
            pen = QPen(Qt.red, 2)
            painter.setPen(pen)
            painter.drawRect(left, top, width, height)

            # 2. Vẽ nền mờ bên trong
            brush = QBrush(QColor(255, 0, 0, 50))  # đỏ, alpha = 50/255
            painter.fillRect(left, top, width, height, brush)

            # 3. Vẽ tâm hình chữ nhật
            cx, cy = left + width // 2, top + height // 2
            pen_center = QPen(Qt.yellow, 4)  # tâm màu vàng
            painter.setPen(pen_center)
            painter.drawPoint(cx, cy)  # hoặc vẽ dấu +

            # Nếu muốn dấu cộng thay vì chấm:
            # painter.drawLine(cx - 5, cy, cx + 5, cy)
            # painter.drawLine(cx, cy - 5, cx, cy + 5)
    
    ## Thêm
    def get_shape(self)-> tuple[str, tuple[int,int], tuple[int,int]]:
        """
        Lấy tên của shape và giá trị start, end để lưu mẫu
        
        """
        if self.start_img and self.end_img:
            # Trả lại theo tọa độ tuyệt đối so với ảnh
            return ("box", self.start_img, self.end_img)
        return None
    