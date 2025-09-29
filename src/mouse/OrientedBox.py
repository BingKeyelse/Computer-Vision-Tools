from libs import*

# ==== Tool cụ thể: vẽ Oriented Box ====
class OrientedBoxTool(MouseTool):
    def __init__(self):
        self.start = None
        self.end = None
        self.angle = 0.0   # góc xoay (radian)
        self.mode = None
        self.offset = (0, 0)

        # Thêm các trạng thái cho resize
        self.base_center = None
        self.base_hw = None
        self.base_hh = None
        self.start_mouse = None
        self.resize_normal = None

        self.start_img= None
        self.end_img= None

    def reset_image(self)-> None:
        """Khởi tạo lại giá trị bắt đầu để reset hình đang vẽ"""
        self.start = None
        self.end = None

    def get_center(self)-> tuple[int, int]:
        """Trả về tâm của hình chữ nhật hiện hiện tại dù nó có xoay hay không"""
        x1, y1 = self.start
        x2, y2 = self.end
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def get_corners(self)-> tuple[(int,int), (int,int), (int,int), (int,int)] | None:
        """Trả về 4 toạ độ của các đỉnh sau khi xoay."""
        if not self.start or not self.end:
            return []
        x1, y1 = self.start
        x2, y2 = self.end

        # Lấy tâm của hình hiện tại dù có xoay hay không
        cx, cy = self.get_center()

        w, h = abs(x2 - x1), abs(y2 - y1)
        # Tọa độ box chuẩn (chưa xoay) so với tâm của hình chữ nhật gốc
        # ------------→ w
        # |
        # |
        # |
        # |
        # ↓ H
        corners = [
            (-w/2, -h/2), # Điểm trên cùng bên trái
            ( w/2, -h/2), # Điểm trên cùng bên phải
            ( w/2,  h/2), # Điểm dưới cùng bên phải
            (-w/2,  h/2), # Điểm dưới cùng bên trái
        ]

        # Xoay ra góc để có các toạ độ đỉnh mới
        cos_a, sin_a = np.cos(self.angle), np.sin(self.angle)
        rotated = []
        for (dx, dy) in corners:
            rx = cx + dx*cos_a - dy*sin_a
            ry = cy + dx*sin_a + dy*cos_a
            rotated.append((rx, ry))
        return rotated

    def get_handle_point(self)-> tuple[int, int] | None:
        """Tìm điểm giữa 2 điểm trên cùng -> dùng để vẽ điểm màu vàng"""
        corners = self.get_corners()
        if not corners:
            return None
        
        # Lấy 2 điểm gần nhất trên cùng 
        x1, y1 = corners[0]
        x2, y2 = corners[1]
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def get_edge_midpoint(self, edge_index: int) -> tuple[int, int] | None:
        """Lấy trung điểm của 1 cạnh bất kỳ trong box.
        edge_index: 0=trên-trái, 1=trên-phải, 2=dưới-phải, 3=dưới-trái
        """
        corners = self.get_corners()
        if not corners or len(corners) != 4:
            return None
        
        i1 = edge_index % 4
        i2 = (edge_index + 1) % 4
        x1, y1 = corners[i1]
        x2, y2 = corners[i2]
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def on_mouse_down(self, event)-> None:
        """Xử lý sự kiện nhấn chuột."""
        # Khởi tạo lúc đầu
        if not self.start or not self.end:
            self.start = (event.x(), event.y())
            self.end = self.start
            return
        
        x, y = event.x(), event.y()
        handle = self.get_handle_point()

        dx = x - handle[0]
        dy = y - handle[1]
        # Bấm gần điểm màu vàng
        if handle and (dx*dx + dy*dy) ** 0.5 < 10:
            self.mode = "rotate"
            cx, cy = self.get_center()
            self.start_angle = np.arctan2(y - cy, x - cx)   # góc ban đầu của chuột so với tâm hình
            self.base_angle = self.angle                    # lưu góc gốc hiện tại của hình 
            return

        # 2. Kiểm tra tâm (move)
        cx, cy = self.get_center()
        if abs(x - cx) < 10 and abs(y - cy) < 10:
            self.mode = "move"
            self.offset = (x - cx, y - cy)
            return
        
        # 3. Kiểm tra midpoint các cạnh (resize)
        for i in range(1,4):   # 4 cạnh
            mid = self.get_edge_midpoint(i) # Lấy trung điểm cạnh đó
            if mid:
                dx, dy = x - mid[0], y - mid[1]
                # Nếu kích vào điểm đó 
                if (dx*dx + dy*dy) ** 0.5 < 10:
                    self.mode = f"resize-edge-{i}"
                    self.active_edge = i # Kích hoạt cạnh i (1->4, Không có 0)
                    
                    # Lấy toạ độ điểm nhấn chuột
                    self.old_mouse = np.array([x, y], dtype=float)

                    # Lấy toạ độ hình lúc đầu
                    cx, cy = self.get_center()
                    self.base_center = np.array([cx, cy], dtype=float)

                    # Lấy kích thước hình chữ nhật khi ở trục song song
                    w = abs(self.end[0] - self.start[0])
                    h = abs(self.end[1] - self.start[1])
                    self.base_hw = w / 2.0 # Từ tâm box ra biên trái/phải
                    self.base_hh = h / 2.0 # Từ tâm box ra biên trên/dưới

                    # normal của cạnh i tại thời điểm bắt đầu
                    corners = self.get_corners()
                    # Tính vector pháp tuyến
                    p1 = np.array(corners[i])
                    p2 = np.array(corners[(i+1)%4])
                    edge_vec = p2 - p1
                    edge_vec = edge_vec / np.linalg.norm(edge_vec)
                    
                    # Đinh hình lại vector pháp tuyến có thể thay đổi tuỳ hường vector của mình
                    self.resize_normal = np.array([edge_vec[1], -edge_vec[0]])
                    return
                
        # Nếu không bấm vào chỗ đó thì khởi tạo
        else:
            self.mode = None
            self.start = (x, y)
            self.end = self.start

    def on_mouse_move(self, event, x_offset, y_offset) -> None:
        """Xử lý sự kiện di chuột."""
        if not self.start or not self.end:
            return
        
        x, y = event.x(), event.y()
        if self.mode == "rotate":
            cx, cy = self.get_center()
            current_angle = np.arctan2(y - cy, x - cx) # góc hiện tại của chuột
            self.angle = self.base_angle + (current_angle - self.start_angle)
        
        elif self.mode == "move":
            # dịch chuyển toàn bộ box
            cx, cy = self.get_center()
            dx, dy = x - cx - self.offset[0], y - cy - self.offset[1]
            self.start = (self.start[0] + dx, self.start[1] + dy)
            self.end   = (self.end[0] + dx, self.end[1] + dy)

        elif self.mode and self.mode.startswith("resize-edge"):
            # Lấy toạ độ điểm hiện tại
            cur = np.array([x, y], dtype=float)

            # Vector từ vị trí chuột bắt đầu resize → vị trí chuột hiện tại
            mvec = cur - self.old_mouse

            # Chiếu mvec lên vector pháp tuyến của cạnh đang resize
                # mvec là vector chuyển động của chuột từ lúc bạn bấm vào midpoint cạnh đó.
                # resize_normal vector pháp tuyến tính ở trên
            delta = float(np.dot(mvec, self.resize_normal))

            if self.active_edge in (0, 2):  # top/bottom -> thay đổi h

                # Cập nhập nửa chiều cao ban đầu
                new_hh = max(4.0, self.base_hh + 0.5 * delta)

                # Cập nhập center 
                # self.resize_normal chỉ là vector pháp tuyến hướng di chuyển
                new_center = self.base_center + 0.5 * delta * self.resize_normal
                self.start = (new_center[0] - self.base_hw, new_center[1] - new_hh) # giữ placeholder
                self.end   = (new_center[0] + self.base_hw, new_center[1] + new_hh)
            else:  # left/right -> thay đổi w
                new_hw = max(4.0, self.base_hw + 0.5 * delta)
                new_center = self.base_center + 0.5 * delta * self.resize_normal
                self.start = (new_center[0] - new_hw, new_center[1] - self.base_hh)
                self.end   = (new_center[0] + new_hw, new_center[1] + self.base_hh)

        else:
            self.end = (x, y)

    def on_mouse_up(self, event)-> None:
        self.mode = None
        self.active_edge = None
        self.start_mouse = None
        self.base_center = None
        self.base_hw = None
        self.base_hh = None
        self.resize_normal = None

    def draw(self, painter, x_offset=0, y_offset=0, ratio_base_image=0)-> None:
        if self.start and self.end:
            corners = self.get_corners()

            # Tính toán để trả về cho thằng get shape
            x1, y1 = self.start
            x2, y2 = self.end

            # print(f'Giá trị offset trên này==== {x_offset} {y_offset}')

            # Trừ offset để ra tọa độ trên ảnh scaled
            self.start_img_scaled = (x1 - x_offset, y1 - y_offset)
            self.end_img_scaled   = (x2 - x_offset, y2 - y_offset)

            # Tọa độ trên ảnh thực tế
            # Đây là bước tính toán để truyền thẳng cho self.get_shape vì tính global của biến
            self.start_img = (self.start_img_scaled[0] / ratio_base_image[0],
                  self.start_img_scaled[1] / ratio_base_image[1])
            self.end_img = (self.end_img_scaled[0] / ratio_base_image[0],
                self.end_img_scaled[1] / ratio_base_image[1])
            
            ##############

            # vẽ polygon vì 4 điểm hiện tại không khác gì polygon
            pen = QPen(Qt.red, 2)
            painter.setPen(pen)
            qpoints = [QPointF(x, y) for (x, y) in corners]
            painter.drawPolygon(QPolygonF(qpoints))

            # nền mờ
            brush = QBrush(QColor(255, 0, 0, 50))
            painter.setBrush(brush)
            painter.drawPolygon(QPolygonF(qpoints))

            # vẽ tâm
            cx, cy = self.get_center()
            painter.setPen(QPen(Qt.yellow, 4))
            painter.drawPoint(int(cx), int(cy))

            # vẽ handle màu vàng
            handle = self.get_handle_point()
            if handle:
                painter.setBrush(QBrush(Qt.yellow))
                painter.drawEllipse(QPointF(*handle), 5, 5)
            
            # Vẽ lần lượt 3 điểm còn lại
            for i in range(1,4):
                center_edge = self.get_edge_midpoint(i)
                if center_edge:
                    painter.setPen(QPen(Qt.red))
                    painter.setBrush(QBrush(Qt.red))
                    painter.drawEllipse(QPointF(*center_edge), 4, 4)


    def get_shape(self)-> None:
        if self.start_img and self.end_img:
            return ("oriented_box", self.start_img, self.end_img, self.angle)
        return None