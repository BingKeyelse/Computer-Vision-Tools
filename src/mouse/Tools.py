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
        self.image= None

    def set_tool(self, tool: MouseTool): # type hint ( gợi ý kiểu)
        """Chọn tool hiện tại để biết sẽ vẽ shape gì Box, Circle, Oriented Box, Polygon."""

        self.active_tool = tool

    def handle_event(self, event_type, event, x_offset=0, y_offset=0, image=None)-> None:
        """Gửi event cho tool đang active. như nhấn, nhả, di chuyển với chuột đang được chọn"""

        self.image = image
        print("ToolManager nhận image:", type(image), getattr(image, "shape", None))

        if not self.active_tool:
            return
        if event_type == "mouse_down":
            self.active_tool.on_mouse_down(event)
        elif event_type == "mouse_move":
            self.active_tool.on_mouse_move(event, x_offset, y_offset)
        elif event_type == "mouse_up":
            self.active_tool.on_mouse_up(event)

    def reset(self)-> None:
        """
        Khi để reset lại hình khi mà nó đang được vẽ
        """
        self.active_tool.reset_image()
    
    def cut(self)-> None:
        """Nhận hình hiện tại và reset tool."""

        if self.active_tool:
            shape = self.active_tool.get_shape()
            
            if shape:
                # Gọi hàm xử lý crop chung
                self.crop_shape(shape)

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
        """Vẽ các hình đã chấp nhận + hình đang thao tác.
        Chuyển đổi vị trí tuyệt đối với ảnh thực tế kích thước thật
        sang vị trí tuyệt đối so với ảnh scaled rồi tiếp tới mới là Widget
        """

        # Xem qua các shape mà mình đã chấp nhận 
        for shape in self.shapes:
            # print(shape)
            if shape[0] == "box":
                _, start, end, idx = shape
                painter.setPen(QPen(Qt.blue, 2))

                # Không thay đổi
                # x1, y1 = start
                # x2, y2 = end
                # Lấy tọa độ

                # Nhận được tọa độ ảnh thực giờ biến sang ảnh scaled
                x1_scaled, y1_scaled = int(start[0] *ratio_base_image[0]), int(start[1] *ratio_base_image[1])
                x2_scaled, y2_scaled = int(end[0] *ratio_base_image[0]), int(end[1] *ratio_base_image[1])
                
                # x1_scaled, y1_scaled = int(start[0] *ratio_base_image[0]*scale_resize), int(start[1] *ratio_base_image[1]*scale_resize)
                # x2_scaled, y2_scaled = int(end[0] *ratio_base_image[0]*scale_resize), int(end[1] *ratio_base_image[1]*scale_resize)
                # print(f"Giá trị nhận được là start({start}) và end({end})")
                # print(f"Giá trị nhận được sau scaled là start({x1_scaled}) và end({y1_scaled})")

                # Thay đổi sang tọa độ tương đối so với Widget
                x1, y1 = x1_scaled + x_offset, y1_scaled + y_offset
                x2, y2 = x2_scaled + x_offset, y2_scaled + y_offset

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
                # x1, y1 = start
                # x2, y2 = end

                # Nhận được tọa độ ảnh thực giờ biến sang ảnh scaled
                x1_scaled, y1_scaled = int(start[0] *ratio_base_image[0]), int(start[1] *ratio_base_image[1])
                x2_scaled, y2_scaled = int(end[0] *ratio_base_image[0]), int(end[1] *ratio_base_image[1])

                # Thay đổi sang tọa độ tương đối so với Widget
                x1, y1 = x1_scaled + x_offset, y1_scaled + y_offset
                x2, y2 = x2_scaled + x_offset, y2_scaled + y_offset


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

                qpoints = []
                for (x_img, y_img) in points:
                    # 1. Ảnh gốc → scaled
                    x_scaled = int(x_img * ratio_base_image[0])
                    y_scaled = int(y_img * ratio_base_image[1])

                    # 2. Scaled → widget
                    x = x_scaled + x_offset
                    y = y_scaled + y_offset

                    qpoints.append(QPoint(x, y))

                # Vẽ polygon
                if len(qpoints) >= 3:  # đủ 3 điểm thì mới khép kín polygon
                    pen = QPen(Qt.blue, 2)
                    brush = QBrush(QColor(0, 0, 255, 40))  # xanh, alpha=40 để mờ
                    painter.setPen(pen)
                    painter.setBrush(brush)
                    painter.drawPolygon(QPolygon(qpoints))

                # # Vẽ từng cạnh (phòng trường hợp polyline chưa khép kín)
                # for i in range(len(qpoints) - 1):
                #     painter.drawLine(qpoints[i], qpoints[i+1])

                # Vẽ ID
                if qpoints:
                    painter.setPen(Qt.yellow)
                    painter.drawText(qpoints[0].x(), qpoints[0].y() - 5, f"ID:{idx}")
                    
            elif shape[0] == "oriented_box":
                _, start, end, angle, idx = shape  # nếu bạn muốn đánh số id giống box

                # Nhận được tọa độ ảnh thực giờ biến sang ảnh scaled
                x1_scaled, y1_scaled = int(start[0] *ratio_base_image[0]), int(start[1] *ratio_base_image[1])
                x2_scaled, y2_scaled = int(end[0] *ratio_base_image[0]), int(end[1] *ratio_base_image[1])

                # Thay đổi sang tọa độ tương đối so với Widget
                x1, y1 = x1_scaled + x_offset, y1_scaled + y_offset
                x2, y2 = x2_scaled + x_offset, y2_scaled + y_offset
            
                # Lấy 4 đỉnh sau khi xoay
                cx, cy = ( (x1+x2)/2, (y1+y2)/2 )
                w, h = abs(x2-x1), abs(y2-y1)
                
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
    
    def crop_shape(self, shape):
        shape_type = shape[0]
        print(shape)

        if shape_type == "box":
            _, start, end = shape
            x1, y1 = map(int, start)
            x2, y2 = map(int, end)
            left, right = sorted([x1, x2])
            top, bottom = sorted([y1, y2])
            # print(self.image.shape)

            cropped = self.image[top:bottom, left:right]
            if cropped.size > 0:
                self.save_cropped_image(cropped)

        elif shape_type == "circle":
            _, start, end, idx = shape
            x1, y1 = map(int, start)
            x2, y2 = map(int, end)
            r = int(((x2 - x1)**2 + (y2 - y1)**2)**0.5)

            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (x1, y1), r, 255, -1)
            cropped = cv2.bitwise_and(self.image, self.image, mask=mask)

            cv2.imwrite(f"./output/circle_{idx}.jpg", cropped)
        elif shape_type=="polygon":
            _, points, idx = shape
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
            cropped = cv2.bitwise_and(self.image, self.image, mask=mask)

            cv2.imwrite(f"./output/polygon_{idx}.jpg", cropped)
        elif shape_type=="oriented_box":
            _, start, end, angle, idx = shape

            # Lấy tâm, w, h
            cx, cy = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            w, h = abs(end[0] - start[0]), abs(end[1] - start[1])

            # Tạo rotated rect trong OpenCV
            rect = ((cx, cy), (w, h), np.degrees(angle))
            box = cv2.boxPoints(rect).astype(np.int32)

            # Crop theo rotated rect
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [box], 255)
            cropped = cv2.bitwise_and(self.image, self.image, mask=mask)

            cv2.imwrite(f"./output/oriented_box_{idx}.jpg", cropped)

    
    def save_cropped_image(self, cropped):
        if cropped.size == 0:
            return

        # Đường dẫn gợi ý mặc định
        default_dir = r"D:\Desktop_with_Data_Pronics\Computer_Vision_Tool\src\data\images"
        default_path = os.path.join(default_dir, f"box.jpg")

        # Hộp thoại chọn nơi lưu + định dạng
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getSaveFileName(
            None,
            "Save Cropped Image",
            default_path,  # gợi ý sẵn
            "JPEG Files (*.jpg);;PNG Files (*.png);;BMP Files (*.bmp)",
            options=options
        )

        if filePath:
            ext = os.path.splitext(filePath)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
                filePath += ".jpg"  # mặc định jpg nếu user quên

            cv2.imwrite(filePath, cropped)
            print(f"Saved to {filePath}")