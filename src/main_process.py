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
    
    def resizeEvent(self, event):
        if self.orig_image:
            # Scale ảnh theo khung label
            self.scaled = self.orig_image.scaled(
                self.width(), self.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # tính toán offset để căn giữa
            self.x_offset = (self.width() - self.scaled.width()) // 2
            self.y_offset = (self.height() - self.scaled.height()) // 2

        # Qt có sẵn event system, mỗi khi widget thay đổi kích thước thì Qt sẽ gọi tự động resizeEvent.
        super().resizeEvent(event)
        

    def cvimg_to_qpixmap(self, cv_img):
        """
        Chuyển ảnh từ OpenCV (numpy array BGR) → QPixmap để hiển thị trong QLabel,
        """
        if cv_img is None:
            return None

        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        return pixmap
    
    def in_image_area(self, x, y):
        if not self.orig_image:
            return False

        return (self.x_offset <= x <= self.x_offset + self.scaled.width() and
                self.y_offset <= y <= self.y_offset + self.scaled.height())
    
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
                self.tool_manager.handle_event("mouse_down", event)
            self.update()

    def mouseMoveEvent(self, event):
        """
        Sự kiện chuột di chuyển.
        """
        if self.in_image_area(event.x(), event.y()):
            if self.tool_manager:        
                self.tool_manager.handle_event("mouse_move", event)
            self.update()

    def mouseReleaseEvent(self, event):
        """
        Sự kiện chuột nhả ra.
        """
        if self.in_image_area(event.x(), event.y()):
            if self.tool_manager:
                self.tool_manager.handle_event("mouse_up", event)
            self.update()

    def paintEvent(self, event):
        """
        Hàm vẽ lại giao diện. 
        Gọi super().paintEvent để QLabel vẽ ảnh gốc trước,
        sau đó ToolManager sẽ vẽ thêm các shape/tool lên trên.
        """
        super().paintEvent(event)
        if self.scaled:
            painter = QPainter(self)
            # vẽ ảnh ở vị trí offset
            painter.drawPixmap(self.x_offset, self.y_offset, self.scaled)

        if self.tool_manager:
            # painter = QPainter(self) # QPainter chính là cây cọ trong Qt,
            # self.tool_manager.draw(painter) # Đưa cọ cho ToolManager để nó vứt cho thằng nào thì vứt để sài
            self.tool_manager.draw(painter, self.x_offset, self.y_offset)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.init_UX_UI()

        image_ori= cv2.imread(r'src\data\images\image.jpg')

        # Tạo canvas và gắn vào QLabel có sẵn trong ui
        self.canvas = Canvas(image_ori, parent=self.ui.screen_main) # Nơi để hiển thị và là nơi thao tác chính
        self.canvas.setGeometry(self.ui.screen_main.rect())  # khớp kích thước với label

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