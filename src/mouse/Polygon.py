from libs import*

# ==== Tool cụ thể: vẽ Polygon ====
class PolygonTool(MouseTool):
    '''
    Tool để sử dụng cho mục đích vẽ hình polygon chọn từng điểm một
    '''
    def __init__(self):
        """Tool để vẽ polygon (đa giác)."""
        self.points = []   # danh sách điểm [(x,y), (x,y), ...]
        self.is_finished = False  # cờ đánh dấu đã chốt polygon chưa
        self.points_image = [] 
        self.points_image_100 = [] 
    
    def reset_image(self)-> None:
        """Khởi tạo lại giá trị bắt đầu để reset hình đang vẽ"""
        self.points = []

    def on_mouse_down(self, event) -> None:
        """
        Thêm điểm mới vào polygon khi click chuột.
        Nếu click gần điểm đầu tiên và số điểm >= 3 thì coi như khép polygon.
        """
        if self.is_finished:
            return

        x, y = event.x(), event.y()
        if self.is_finished == False:
            if self.points:
                # Kiểm tra click gần điểm đầu tiên => khép polygon
                x0, y0 = self.points[0]
        
            if len(self.points) >= 3 and abs(x - x0) < 10 and abs(y - y0) < 10:
                self.is_finished = True
                # self.points.append((x0, y0))
            else:
                self.points.append((x, y))
        return

    def on_mouse_move(self, event, x_offset, y_offset) -> None:
        """Có thể vẽ preview đường tạm từ điểm cuối đến chuột."""
        pass  # tuỳ bạn muốn có preview không (nối điểm cuối tới chuột hiện tại)

    def on_mouse_up(self, event) -> None:
        """Không cần gì đặc biệt."""
        pass

    def undo_point(self) -> None:
        """Xoá điểm cuối cùng khi đang vẽ."""
        # if self.points and not self.is_finished:
        if self.points:
            self.points.pop()
            self.is_finished= False

    def get_shape(self):
        """Trả polygon khi đã hoàn tất (is_finished=True)."""
        if self.is_finished and len(self.points_image_100) >= 3:
            self.is_finished=False
            return ("polygon", self.points_image_100)
        return None

    def draw(self, painter, x_offset=0, y_offset=0, ratio_base_image=0, scale=1.0) -> None:
        """Vẽ polygon đang thao tác."""
        if not self.points:
            return

        pen = QPen(Qt.red, 2)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(255, 0, 0, 50)))  # nền mờ

        # nếu đã khép kín
        if self.is_finished:
            # Tính toán lại để gửi về phần get shape
            self.points_image = [] 
            self.points_image_100= []
            for (x, y) in self.points:
                # Trừ offset để ra tọa độ trên ảnh scaled
                x_scaled = x - x_offset
                y_scaled = y - y_offset

                # Đưa về tọa độ trên ảnh gốc (chia theo tỉ lệ)
                x_img = x_scaled / ratio_base_image[0]
                y_img = y_scaled / ratio_base_image[1]

                # Đưa về tọa độ trên ảnh 100%
                x_img_100= x_img/scale
                y_img_100= y_img/scale


                self.points_image.append((x_img, y_img))
                self.points_image_100.append((x_img_100, y_img_100))

            polygon = QPolygon([QPoint(x, y) for x, y in self.points])
            painter.drawPolygon(polygon)
        else:
            # vẽ polyline chưa đóng
            for i in range(len(self.points) - 1):
                painter.drawLine(self.points[i][0], self.points[i][1],
                                 self.points[i+1][0], self.points[i+1][1])
            # vẽ điểm đầu + điểm cuối
            for (x, y) in self.points:
                painter.drawEllipse(QPoint(x, y), 3, 3)