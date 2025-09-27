from libs import*

# ==== Quản lý tool ====
class ToolManager:
    '''
    Tool Manager là công cụ quản lý toàn bộ các tool bao gồm: Box, Circle, Oriented Box, Polygon
    '''
    def __init__(self):
        self.active_tool = None
        ## Thêm
        self.shapes = []  # chứa các hình đã cắt (chấp nhận)
        self.counter = 0   # để đánh số id

    def set_tool(self, tool: MouseTool): # type hint ( gợi ý kiểu)
        """Chọn tool hiện tại (có thể là None)."""

        self.active_tool = tool

    def handle_event(self, event_type, event, x_offset=0, y_offset=0, scaled=0.0)-> None:
        """Gửi event cho tool đang active."""

        if not self.active_tool:
            return
        if event_type == "mouse_down":
            self.active_tool.on_mouse_down(event)
        elif event_type == "mouse_move":
            self.active_tool.on_mouse_move(event, x_offset, y_offset, scaled)
        elif event_type == "mouse_up":
            self.active_tool.on_mouse_up(event)
    
    def cut(self)-> None:
        """Chấp nhận hình hiện tại và reset tool."""

        if self.active_tool:
            shape = self.active_tool.get_shape()
            if shape:
                self.counter += 1
                shape_with_id = shape + (self.counter,)
                self.shapes.append(shape_with_id)
            # reset tool để chuẩn bị vẽ mới
            self.active_tool = type(self.active_tool)()

    def clear(self)-> None:
        """Xoá toàn bộ hình"""

        self.shapes.clear()
        self.counter=0
    
    def undo(self)-> None:
        """Xoá hình cuối cùng"""

        if self.shapes:
            self.shapes.pop()
            self.counter -= 1
    
    def undo_polygon(self)-> None:
        """
        Lệnh này giành riêng cho polygon thôi
        """
        if isinstance(self.active_tool, PolygonTool):
            self.active_tool.undo_point()

    def draw(self, painter, x_offset=0, y_offset=0, ratio_base_image=0)-> None:
        """Vẽ các hình đã chấp nhận + hình đang thao tác."""
        # ox, oy = x_offset, y_offset  # offset khi căn giữa ảnh
        print(f'Giá trị offset trên này+++++++++ {x_offset} {y_offset}')
        for shape in self.shapes:
            print(shape)
            if shape[0] == "box":
                _, start, end, idx = shape
                painter.setPen(QPen(Qt.blue, 2))
                # Không thay đổi
                # x1, y1 = start
                # x2, y2 = end

                # Nhận được tọa độ ảnh thực giờ biến sang ảnh scaled
                x1_scaled, y1_scaled = int(start[0] *ratio_base_image[0]), int(start[1] *ratio_base_image[1])
                x2_scaled, y2_scaled = int(end[0] *ratio_base_image[0]), int(end[1] *ratio_base_image[1])


                # Thay đổi sang tọa độ tương đối so với Widget
                x1, y1 = x1_scaled + x_offset, y1_scaled + y_offset
                x2, y2 = x2_scaled + x_offset, y2_scaled + y_offset

                # print(f"Giá trị thực sự của điểm này sau biến đổi lần lượt là {(x1,y1)} và {(x2,y2)}")

                painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                # vẽ id cạnh hình
                painter.setPen(Qt.yellow)
                painter.drawText(int(x1), int(y1) - 5, f"ID:{idx}")

                # Vẽ lớp phủ
                left, top = min(x1, x2), min(y1, y2)
                width, height = abs(x2 - x1), abs(y2 - y1)
                brush = QBrush(QColor(0, 0, 255, 40))  # blue, alpha=40/255
                painter.fillRect(left, top, width, height, brush)
                
            elif shape[0] == "circle":
                _, start, end, idx = shape
                painter.setPen(QPen(Qt.blue, 2))
                x1, y1 = start
                x2, y2 = end
                # Tính bán kính = khoảng cách từ start đến end
                r = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
                painter.drawEllipse(x1 - r, y1 - r, 2*r, 2*r)
                # vẽ id cạnh hình
                painter.setPen(Qt.yellow)
                painter.drawText(x1, y1 - r - 5, f"ID:{idx}")

                # Vẽ lớp phủ lên
                brush = QBrush(QColor(0, 0, 255, 40))  # blue, alpha=40/255
                painter.setBrush(brush)
                painter.setPen(Qt.NoPen)  # bỏ viền khi fill
                painter.drawEllipse(x1 - r, y1 - r, 2 * r, 2 * r)

            elif shape[0] == "polygon":
                _, points, idx = shape
                painter.setPen(QPen(Qt.blue, 2))
                for i in range(len(points)):
                    x1, y1 = points[i]
                    x2, y2 = points[(i+1) % len(points)]
                    painter.drawLine(x1, y1, x2, y2)

                    # đủ 3 điểm thì coi như polygon
                    pen = QPen(Qt.blue, 2)
                    brush = QBrush(QColor(0, 0, 255, 20))  # xanh, alpha=50 để mờ
                    painter.setPen(pen)
                    painter.setBrush(brush)

                    qpoints = [QPoint(x, y) for x, y in points]
                    painter.drawPolygon(QPolygon(qpoints))

                    # vẽ ID
                    painter.setPen(Qt.yellow)  # ID màu đen cho dễ đọc
                    painter.drawText(points[0][0], points[0][1] - 5, f"ID:{idx}")
            elif shape[0] == "oriented_box":
                _, start, end, angle, idx = shape  # nếu bạn muốn đánh số id giống box
                # Lấy 4 đỉnh sau khi xoay
                cx, cy = ( (start[0]+end[0])/2, (start[1]+end[1])/2 )
                w, h = abs(end[0]-start[0]), abs(end[1]-start[1])
                
                # Tọa độ box chuẩn (chưa xoay) so với tâm
                corners = [
                    (-w/2, -h/2),
                    ( w/2, -h/2),
                    ( w/2,  h/2),
                    (-w/2,  h/2),
                ]

                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rotated = []
                for dx, dy in corners:
                    rx = cx + dx*cos_a - dy*sin_a
                    ry = cy + dx*sin_a + dy*cos_a
                    rotated.append((rx, ry))
                
                # Vẽ polygon
                painter.setPen(QPen(Qt.blue, 2))
                qpoints = [QPointF(x, y) for x, y in rotated]
                painter.drawPolygon(QPolygonF(qpoints))

                # Vẽ nền mờ
                brush = QBrush(QColor(0, 0, 255, 40))
                painter.setBrush(brush)
                painter.drawPolygon(QPolygonF(qpoints))

                # Vẽ ID
                painter.setPen(Qt.yellow)
                painter.drawText(int(cx), int(cy) - 5, f"ID:{idx}")


        # vẽ hình đang thao tác (chưa cut)
        if self.active_tool:
            self.active_tool.draw(painter, x_offset, y_offset, ratio_base_image)