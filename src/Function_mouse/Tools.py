from libs import*

# ==== Quản lý tool ====
class ToolManager:
    '''
    ## Tool Manager là công cụ quản lý toàn bộ các tool bao gồm: Box, Circle, Oriented Box, Polygon
    '''
    def __init__(self):
        self.active_tool = None
        ## Thêm
        self.shapes = []  # chứa các hình đã cắt (chấp nhận)
        self.counter = 0   # để đánh số id
        self.image= None # chứa ảnh resize
        self.link_image=None # chứa link ảnh để làm việc với ảnh gốc

        self.scale_resize=1.0 # tỉ lệ ảnh resize mình muốn truyền

    def set_tool(self, tool: MouseTool): # type hint ( gợi ý kiểu)
        """Chọn tool hiện tại để biết sẽ vẽ shape gì Box, Circle, Oriented Box, Polygon. --> self.active_tool"""

        self.active_tool = tool

    def handle_event(self, event_type, event, x_offset=0, y_offset=0, image_original=None, link_image= None)-> None:
        """
        ## Gửi event cho tool đang active. 
        - Như nhấn, nhả, di chuyển với chuột đang được chọn
        - input
            - event_type: tên kiểu sự kiện
            - event
            - x_offset, y_offset: offset theo cả 2 chiều x, y
            - image_original: hình ảnh gốc đã được resize
            - link_image: link ảnh gốc 
        """

        self.image = image_original
        self.link_image= link_image

        # print("ToolManager nhận image:", type(image_original), getattr(image_original, "shape", None)) #Log

        if not self.active_tool:
            return
        if event_type == "mouse_down":
            self.active_tool.on_mouse_down(event)
        elif event_type == "mouse_move":
            self.active_tool.on_mouse_move(event)
        elif event_type == "mouse_up":
            self.active_tool.on_mouse_up(event)

    def reset(self)-> None:
        """
        ## Để reset khi đang vẽ hình và muốn xóa nó
        - Reset giá trị start, end ở từng thằng đang active về None
        """
        self.active_tool.reset_image()
    
    def cut(self, list_shape= None)-> list:
        """
        ## Nhận hình hiện tại và reset tool.
        ## Lấy dữ kiện đã được cut về trả về cho thằng Sample làm việc
        Lưu ý: toàn bộ kích thước là với ảnh nằm ở dạng size 100% nhé 
        - input: 
            - list_shape: mảng để dùng lưu trữ cho ông Sample
        - output:
            - list_shape: {mode, link_picture, data}
        """

        if self.active_tool:
            shape = self.active_tool.get_shape()
            print(shape)
            
            if shape:
                # Gọi hàm xử lý crop chung
                # self.crop_shape(shape)
                if list_shape is not None:
                    list_shape.append({
                        'mode': 0,
                        'link': self.link_image,
                        'data': shape
                    })

                self.counter += 1
                shape_with_id = shape + (self.counter,)
                self.shapes.append(shape_with_id)
            # reset tool để chuẩn bị vẽ mới
            self.active_tool = type(self.active_tool)()

        if list_shape is not None:
            return list_shape

    def clear(self)-> None:
        """Xoá toàn bộ hình đã đưuọc cut và lưu"""
        self.shapes.clear()
        self.counter=0
    
    def undo(self)-> None:
        """Xoá hình cuối cùng được cut ở cuối cùng"""

        if self.shapes:
            self.shapes.pop()
            self.counter -= 1
    
    # def remove_SHAPE(self, idx=None):
    #     """
    #     ## Dùng để xóa hình ảnh shape ở vị trí bất kì mình chỉ định
    #     - input 
    #         - idx: vị trí trong mảng mình muốn xóa
    #     """
    #     if idx is not None:
    #         self.shapes.pop(idx)
    #         self.counter -= 1
    
    def undo_polygon(self)-> None:
        """
        Lệnh này giành riêng cho polygon thôi dùng để undo điểm đang vẽ
        """
        if isinstance(self.active_tool, PolygonTool):
            self.active_tool.undo_point()

    def draw(self, painter, x_offset=0, y_offset=0, ratio_base_image=0, scale=1.0)-> None:
        """
        ## Vẽ các hình đã chấp nhận + hình đang thao tác.
        - Chuyển đổi vị trí của:  anh size 100% --> ảnh resized đang được dùng
        - Chuyển đổi vị trí của : ảnh resized đang được dùng --> ảnh được scaled
        - Chuyển đổi vị trí của : ảnh được scaled --> offset vị trí tuyệt đối với Widget 
        - input
            - painter: bút lông để vẽ
            - x_offset, y_offset: offset ảnh so với Widget
            - ratio_base_image: tỉ lệ ảnh khi scaled
            - scale: thực ra cái này là tỉ lệ resize mình chọn 
        """

        # Lấy tỉ lệ resize
        self.scale_resize= scale

        # Xem qua các shape mà mình đã chấp nhận 
        for shape in self.shapes:
            # print(shape)
            if shape[0] == "box":
                _, start, end, idx = shape
                painter.setPen(QPen(Qt.blue, 2))
                # print(start)

                # Không thay đổi
                # x1, y1 = start
                # x2, y2 = end
                # Lấy tọa độ

                # Nhận được tọa độ ảnh 100% sang ảnh resize
                x1_resized, y1_resized = int(start[0] *self.scale_resize), int(start[1] *self.scale_resize)
                x2_resized, y2_resized = int(end[0] *self.scale_resize), int(end[1] *self.scale_resize)


                # Nhận được tọa độ ảnh thực resized giờ biến sang ảnh scaled
                x1_scaled, y1_scaled = int(x1_resized *ratio_base_image[0]), int(y1_resized *ratio_base_image[1])
                x2_scaled, y2_scaled = int(x2_resized *ratio_base_image[0]), int(y2_resized *ratio_base_image[1])
                
                # print(f"Giá trị nhận được là start({start}) và end({end})")
                # print(f"Giá trị nhận được sau scaled với tọa độ start({x1_resized},{y1_resized})")

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
                _, start, end, angle, idx = shape                
                # x1, y1 = start
                # x2, y2 = end

                # Nhận được tọa độ ảnh 100% sang ảnh resize
                x1_resized, y1_resized = int(start[0] *self.scale_resize), int(start[1] *self.scale_resize)
                x2_resized, y2_resized = int(end[0] *self.scale_resize), int(end[1] *self.scale_resize)

                # Nhận được tọa độ ảnh thực giờ biến sang ảnh scaled
                x1_scaled, y1_scaled = int(x1_resized *ratio_base_image[0]), int(y1_resized *ratio_base_image[1])
                x2_scaled, y2_scaled = int(x2_resized *ratio_base_image[0]), int(y2_resized *ratio_base_image[1])

                # Thay đổi sang tọa độ tương đối so với Widget
                x1, y1 = x1_scaled + x_offset, y1_scaled + y_offset
                x2, y2 = x2_scaled + x_offset, y2_scaled + y_offset


                # Tính bán kính = khoảng cách từ start đến end
                r = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)

                # === Vẽ hình tròn xanh ===
                painter.setPen(QPen(Qt.blue, 2))
                painter.drawEllipse(x1 - r, y1 - r, 2*r, 2*r)

                

                # Vẽ lớp phủ lên
                brush = QBrush(QColor(0, 0, 255, 40))  # blue, alpha=40/255
                painter.setBrush(brush)
                painter.setPen(Qt.NoPen)  # bỏ viền khi fill
                painter.drawEllipse(x1 - r, y1 - r, 2 * r, 2 * r)

                # === Vẽ hướng và handler ===
                # hx, hy = self.get_handle_pos()
                hx, hy = CircleTool().get_handle_pos_virtual((x1, y1), (x2, y2), angle)
                hx, hy = int(hx), int(hy)

                painter.setPen(QPen(Qt.green, 1))
                painter.drawLine(int(x1), int(y1), hx, hy)

                painter.setBrush(QBrush(Qt.green))
                painter.drawEllipse(QPointF(hx, hy), 4, 4)

                # === Hiển thị góc ===
                painter.setPen(QPen(Qt.white))
                painter.drawText(hx + 10, hy, f"{int(angle)}°")

                # vẽ id cạnh hình
                painter.setPen(Qt.yellow)
                painter.drawText(x1, y1 - r - 5, f"ID:{idx}")
                painter.setBrush(Qt.NoBrush)



            elif shape[0] == "polygon":
                _, points, idx = shape
                painter.setPen(QPen(Qt.blue, 2))

                qpoints = []
                for (x_img_100, y_img_100) in points:
                    # 2. Ảnh size 100 → ảnh gốc
                    x_resized = int(x_img_100 * scale)
                    y_resized = int(y_img_100 * scale)

                    # 2. Ảnh gốc → scaled
                    x_scaled = int(x_resized * ratio_base_image[0])
                    y_scaled = int(y_resized * ratio_base_image[1])

                    # 3. Scaled → widget
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
            self.active_tool.draw(painter, x_offset, y_offset, ratio_base_image, self.scale_resize)
    
    def crop_shape(self, link='', shape= None)-> np.ndarray:
        """
        ## Dùng chỉ để cắt ảnh với các shape mà mình định nghĩa
        - input
            - link: đường dẫn của link ảnh
            - shape: giá trị data mà mình sẽ dùng để xử lý gồm {mode, link_picture, data}
        - output:
            - cropped_masked: ảnh được cut
        """
        if link=='' and not shape:
            return None
        
        shape_type = shape[0]

        image_100= cv2.imread(link)
    
        if shape_type == "box":
            _, start, end = shape
            x1, y1 = map(int, start)
            x2, y2 = map(int, end)
            left, right = sorted([x1, x2])
            top, bottom = sorted([y1, y2])
            # print(image_100.shape)

            cropped_masked = image_100[top:bottom, left:right]
            if cropped_masked.size > 0:
                self.save_cropped_image(cropped_masked)

        elif shape_type == "circle":
            _, start, end, angle = shape
            x1, y1 = map(int, start)
            x2, y2 = map(int, end)

            # bán kính
            r = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            margin = 5
            extra = 2  # thêm 2px để không sát
            R = r + margin + extra

            # tọa độ vùng crop vuông
            x1_crop, y1_crop = max(0, x1 - R), max(0, y1 - R)
            x2_crop, y2_crop = min(image_100.shape[1], x1 + R), min(image_100.shape[0], y1 + R)

            # crop vùng vuông
            cropped = image_100[y1_crop:y2_crop, x1_crop:x2_crop].copy()
            h, w = cropped.shape[:2]

            # tạo mask hình tròn (center = giữa crop)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (w // 2, h // 2), r, 255, -1)

            # giữ phần hình tròn, ngoài vùng là đen
            cropped_masked = cv2.bitwise_and(cropped, cropped, mask=mask)

            if cropped_masked.size > 0:
                self.save_cropped_image(cropped_masked)

        elif shape_type=="polygon":
            _, points = shape
            points = np.array(points, dtype=np.int32)

            # bounding box
            x, y, w, h = cv2.boundingRect(points)

            # thêm margin
            margin = 10
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, image_100.shape[1])
            y2 = min(y + h + margin, image_100.shape[0])

            # crop ảnh gốc
            cropped = image_100[y1:y2, x1:x2].copy()

            # tạo mask cùng kích thước crop
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)

            # dịch polygon về hệ tọa độ crop
            points_shifted = points - [x1, y1]
            points_shifted = points_shifted.astype(np.int32)
            points_shifted = points_shifted.reshape((-1, 1, 2))   # rất quan trọng
            cv2.fillPoly(mask, [points_shifted], 255)

            # áp mask: giữ nguyên trong polygon, ngoài = đen
            cropped_masked = cv2.bitwise_and(cropped, cropped, mask=mask)

            # cv2.imwrite("debug_cropped.png", cropped)
            # cv2.imwrite("debug_mask.png", mask)
            # cv2.imwrite("debug_result.png", cropped_masked)

            if cropped_masked.size > 0:
                self.save_cropped_image(cropped_masked)


        elif shape_type=="oriented_box":
            _, start, end, angle = shape

            # Lấy tâm, w, h
            cx, cy = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            w, h = abs(end[0] - start[0]), abs(end[1] - start[1])

            # Tạo rotated rect trong OpenCV
            rect = ((cx, cy), (w, h), np.degrees(angle))
            box = cv2.boxPoints(rect).astype(np.int32)

            # Tính bounding box quanh rotated rect
            x, y, w_box, h_box = cv2.boundingRect(box)

            # Crop ảnh gốc
            cropped = image_100[y:y+h_box, x:x+w_box].copy()

            # Tạo mask kích thước như crop
            mask = np.zeros((h_box, w_box), dtype=np.uint8)

            # Dịch box về toạ độ crop
            box_shifted = box - [x, y]
            cv2.fillPoly(mask, [box_shifted], 255)

            # Áp mask → trong rotated rect giữ nguyên, ngoài = đen
            cropped_masked = cv2.bitwise_and(cropped, cropped, mask=mask)

            if cropped_masked.size > 0:
                self.save_cropped_image(cropped_masked)
        return cropped_masked

    def save_cropped_image(self, cropped):
        """
        ## Dùng để save ảnh được cut với chỉ định folder nhé
        - input
            - cropped: ảnh muốn lưu
        """
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