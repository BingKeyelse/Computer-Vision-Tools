from libs import*

class Canvas(QLabel):
    def __init__(self, img, parent=None):
        """
        Canvas kế thừa từ QLabel để vừa hiển thị ảnh, 
        vừa là vùng vẽ + xử lý sự kiện chuột cho các Tool.
        """
        super().__init__(parent)
        self.orig_image  = self.cvimg_to_qpixmap(img)
        # self.setPixmap(self.orig_image)
        # self.setScaledContents(True)  # tự động fit nội dung theo khung label
        self.tool_manager = None
        self.scale_paramater=(0,0)
    
    def resizeEvent(self, event):
        """
        Hàm event handler có sẵn của Qt
        Khi widget đổi kích thước (ví dụ bạn kéo giãn cửa sổ), Qt tự động gọi hàm này.
        Ta override để cập nhật lại cách hiển thị ảnh.
        """
        if self.orig_image:
            # Scale ảnh theo kích thước của widget giờ ta có ảnh đã scale
            self._image_scaled = self.orig_image.scaled(
                self.width(), self.height(),
                Qt.KeepAspectRatio, # giữ nguyên tỷ lệ khung hình (không méo ảnh).
                Qt.SmoothTransformation # dùng thuật toán nội suy mượt (chậm hơn nhưng chất lượng tốt).
            )

            # Tính toán offset để căn giữa vì chưa chắc 2 widget và ảnh đã fix với nhau
            # Là số pixel dịch để căn giữa ảnh trong widget, không phải tỉ lệ.
            # Do lệnh Qt.KeepAspectRatio
            self.x_offset = (self.width() - self._image_scaled.width()) // 2
            self.y_offset = (self.height() - self._image_scaled.height()) // 2
            print(f'Giá trị offset {self.x_offset}  {self.y_offset}')

            # Tính tỉ lệ so với ảnh gốc 
            self.ratio_base_image = (self._image_scaled.width() / self.orig_image.width(), self._image_scaled.height() / self.orig_image.height())

            # Sau khi có tỉ lệ này muốn suy ra tọa độ điểm ảnh trên ảnh gốc. Theo cthuc bên dưới
            # x_img = (event.x() - x_offset) / ratio
            # y_img = (event.y() - y_offset) / ratio

        # Qt có sẵn event system, mỗi khi Widget thay đổi sẽ gọi tự động resizeEvent.
        super().resizeEvent(event)
        

    def cvimg_to_qpixmap(self, cv_img):
        """
        Chuyển ảnh từ OpenCV (numpy array BGR) → QPixmap để hiển thị trong QLabel
        Bởi vì Qlabel cần là đối tượng ở dạng QPixmap hoặc QImage
        """
        if cv_img is None:
            return None

        # Qt dùng chuẩn vơi RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        # Tính toán để chuyển
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        return pixmap
    
    def in_image_area(self, x, y):
        if not self.orig_image:
            return False

        return (self.x_offset <= x <= self.x_offset + self._image_scaled.width() and
                self.y_offset <= y <= self.y_offset + self._image_scaled.height())
    
    def set_tool_manager(self, manager):
        """
        Gắn ToolManager cho canvas. 
        Nếu manager = None thì không làm gì cả.
        """
        self.tool_manager = manager

    def mousePressEvent(self, event):
        """
        Sự kiện chuột nhấn. 
        Nếu tool_manager = None thì return None (bỏ qua).
        """
        if self.in_image_area(event.x(), event.y()):
            if self.tool_manager:
                self.tool_manager.handle_event("mouse_down", event, self.x_offset, self.y_offset)
            self.update()

    def mouseMoveEvent(self, event):
        """
        Sự kiện chuột di chuyển.
        """
        if self.in_image_area(event.x(), event.y()):
            if self.tool_manager:        
                self.tool_manager.handle_event("mouse_move", event, self.x_offset, self.y_offset, self.scale_paramater)
            self.update()

    def mouseReleaseEvent(self, event):
        """
        Sự kiện chuột nhả ra.
        """
        if self.in_image_area(event.x(), event.y()):
            if self.tool_manager:
                self.tool_manager.handle_event("mouse_up", event, self.x_offset, self.y_offset)
            self.update()

    def paintEvent(self, event):
        """
        Hàm vẽ lại giao diện. 
        Gọi super().paintEvent để QLabel vẽ ảnh gốc trước,
        sau đó ToolManager sẽ vẽ thêm các shape/tool lên trên.
        paintEvent sẽ tự động chạy lại khi 
            widget được hiển thị lần đầu
            widget bị che/hiện lại
            widget thay đổi kích thước
            gọi update() hoặc repaint()
            widget nội bộ bị đổi (ví dụ ảnh mới, màu nền mới)
        """ 
        super().paintEvent(event)
        # print('Có thay đổi theo')
        if self._image_scaled:
            painter = QPainter(self)

            # vẽ lại ảnh ở vị trí offset và với kích thước được scale
            painter.drawPixmap(self.x_offset, self.y_offset, self._image_scaled)

        if self.tool_manager:
            # painter = QPainter(self) # QPainter chính là cây cọ trong Qt,
            # self.tool_manager.draw(painter) # Đưa cọ cho ToolManager để nó vứt cho thằng nào thì vứt để sài
            self.tool_manager.draw(painter, self.x_offset, self.y_offset, self.ratio_base_image)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.init_UX_UI()

        image_ori= cv2.imread(r'src\data\images\image.jpg')

        # Tạo canvas và gắn vào QLabel có sẵn trong ui. Chúng nằm đè lên chứ không phải là một
        self.canvas = Canvas(image_ori, parent=self.ui.screen_main) #  QLabel (Canvas) = Widget: để hiển thị và là nơi thao tác chính

        # How to debug with widget to set fix with label
        # geometry() Thì lấy theo tọa độ của cha
        # rect(): Thì lấy theo tọa độ cục bộ
        # self.canvas.setGeometry(self.ui.screen_main.geometry())  # khớp kích thước với label
        # self.canvas.setGeometry(self.ui.screen_main.rect())
        # self.canvas.setStyleSheet("background-color: rgba(255, 0, 0, 100);") 

        # print("screen_main size:", self.ui.screen_main.width(), self.ui.screen_main.height())
        # print("canvas size:", self.canvas.width(), self.canvas.height())
        # QTimer.singleShot(100, lambda: print(
        #     "AFTER SHOW:",
        #     self.ui.screen_main.width(), self.ui.screen_main.height(),
        #     self.canvas.width(), self.canvas.height()
        # ))

        layout = QVBoxLayout(self.ui.screen_main) # Gắn 1 layout để quản lý toàn bộ thông tin bên trong của label
        layout.setContentsMargins(0,0,0,0) # Bỏ hết viền (margin) không sẽ để mặc định là 9
        layout.addWidget(self.canvas) # Bỏ canvas vào nó sẽ chiếm toàn bộ không gian trong label
        # QTimer.singleShot(100, lambda: print(
        #     "AFTER SHOW:",
        #     self.canvas.scale_paramater
        # ))

        
        self.tool_manager = ToolManager() # Đạo diễn, người chỉ định dùng tool nào
        self.canvas.set_tool_manager(self.tool_manager)

        # Chọn tool mặc định là Box
        self.tool_manager.set_tool(BoxTool())

        self.ui.btn_shape.currentTextChanged.connect(self.change_tool)

        self.ui.btn_cut.clicked.connect(lambda: (self.tool_manager.cut(), self.canvas.update()))
        self.ui.btn_clear.clicked.connect(lambda: (self.tool_manager.clear(), self.canvas.update()))
        self.ui.btn_undo.clicked.connect(lambda: (self.tool_manager.undo(), self.canvas.update()))
        self.ui.btn_polyundo.clicked.connect(lambda: (self.tool_manager.undo_polygon(), self.canvas.update()))
    
    def init_UX_UI(self):
        self.ui.btn_polyundo.hide()

    
    def change_tool(self, tool_name):
        self.ui.btn_polyundo.hide()
        if tool_name == "Box":
            self.tool_manager.set_tool(BoxTool())
        elif tool_name == "Circle":
            self.tool_manager.set_tool(CircleTool())
        elif tool_name == "Polygon":
            self.tool_manager.set_tool(PolygonTool())
            self.ui.btn_polyundo.show()
        elif tool_name == "Oriented Box":
            self.tool_manager.set_tool(OrientedBoxTool())
       

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())