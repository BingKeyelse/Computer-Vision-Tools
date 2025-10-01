from libs import*

class ButtonController:
    def __init__(self, ui, tool_manager, canvas):
        """
        Chứa tất cả các hàm liên quan đến Button
        """
        self.ui = ui
        self.tool_manager= tool_manager
        self.canvas= canvas

        # Function Camera
        self.image=None
        self.file_path=None
        self.ui.btn_resize.currentTextChanged.connect(self.resize_image) 

        # Function ToolBar
        self.link_picutures= []
        self.ui.btn_open.clicked.connect(self.get_link_image)
        # self.ui.list_image.setContextMenuPolicy(Qt.CustomContextMenu) # Không dùng context mặc định mà dùng dạng custom
        # self.ui.list_image.customContextMenuRequested.connect(self.show_list_menu)# Khi bạn bấm chuột phải vào thì phát singal tới slot được định, và auto truyền pos
        
        # Function Tool Shape
        self.ui.btn_shape.currentTextChanged.connect(self.change_tool) #Singal tự gửi được Toolname của QListWidget
        self.ui.btn_cut.clicked.connect(lambda: (self.tool_manager.cut(), self.canvas.update()))
        self.ui.btn_clear.clicked.connect(lambda: (self.tool_manager.clear(), self.canvas.update()))
        self.ui.btn_undo.clicked.connect(lambda: (self.tool_manager.undo(), self.canvas.update()))
        self.ui.btn_polyundo.clicked.connect(lambda: (self.tool_manager.undo_polygon(), self.canvas.update()))
        self.ui.btn_new.clicked.connect(lambda: (self.tool_manager.reset(), self.canvas.update()))

        for i in range(1, 12):
            btn = getattr(self.ui, f"btn_sample_{i}", None)
            if btn:
                btn.clicked.connect(lambda _, idx=i: self.handle_sample_click(idx))
                
    def handle_sample_click(self, idx):
        print(f"Clicked index: {idx}")

    def  resize_image(self, size_text)-> None:
        """ Dùng để lựa chọn để tùy chỉnh kích thước của ảnh để cho nó phù hợp với chương trình chạy"""

        # Reset toàn bộ data trước khi vẽ lên mới nhé
        # self.tool_manager.clear()
        # self.tool_manager.reset()

        scale= int(size_text)/100

        if self.image is None:
            return None
    
        h, w = self.image.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w <= 0 or new_h <= 0:
            return None  # Tránh resize về 0

        # Resize ảnh bằng OpenCV
        resized = cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Lưu lại ảnh đã resize (nếu bạn muốn giữ)
        self.image_resized = resized  
        self.get_information_image(self.image_resized)

        # Nếu bạn đang có canvas để hiển thị thì update luôn
        if hasattr(self, "canvas"):
            self.canvas.set_image(resized, self.file_path, scale)
        
    def change_tool(self, tool_name)-> None:
        """
        Hàm lựa chọn để thay đổi các shape tool cho ToolManager
        """
        
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

    def get_link_image(self):
        """Mở hộp thoại chọn file ảnh.
        Nếu chọn thì ảnh sẽ vào ListWidget  theo dạng insertItem
        . Xong show ảnh luôn
        """

        self.file_path, _ = QFileDialog.getOpenFileName(
            None, # Nếu ở trong MainWindow thì truyền self
            "Chọn ảnh",                # tiêu đề hộp thoại
            "",                        # thư mục mặc định ("" = thư mục hiện tại)
            "Image Files (*.png *.jpg *.jpeg *.bmp)"  # filter chỉ cho phép chọn ảnh
        )
        if self.file_path:  # Nếu user chọn ảnh (không bấm Cancel)
            # print("Ảnh được chọn:", self.file_path)
            self.link_picutures.append(self.file_path)
            # Hiển thị lên QListWidget
            self.ui.list_image.insertItem(0,self.file_path)  # listWidget là QListWidget trong .ui của bạn

             # Nếu số lượng item > 10 thì xóa item cuối
            if self.ui.list_image.count() > 10:
                self.ui.list_image.takeItem(self.ui.list_image.count() - 1)

        # Show ảnh luôn        
        self.choose_selected_item()
        return None
    
    def show_list_menu(self, pos):
        # Lấy widget gọi (list_image)
        widget = self.ui.list_image  

        # Tạo menu context gắn parent là QListWidget
        menu = QMenu(widget)

        # Thêm action chọn ảnh xử lý
        choose_action = QAction("Choose", widget)
        choose_action.triggered.connect(self.choose_selected_item)
        menu.addAction(choose_action)

        # Thêm action Xóa
        delete_action = QAction("Delete", widget)
        delete_action.triggered.connect(self.delete_selected_item)
        menu.addAction(delete_action)

        # Thêm action Copy
        copy_action = QAction("Copy link path", widget)
        copy_action.triggered.connect(self.copy_selected_item)
        menu.addAction(copy_action)
        
        # Show menu ngay tại vị trí click
        menu.exec_(widget.mapToGlobal(pos))


    def copy_selected_item(self):
        item = self.ui.list_image.currentItem()
        if item:  # tránh lỗi nếu chưa chọn gì
            QApplication.clipboard().setText(item.text())

    def delete_selected_item(self):
        item = self.ui.list_image.currentItem()
        if item:
            row = self.ui.list_image.row(item)
            self.ui.list_image.takeItem(row)
    
    def choose_selected_item(self):
        """
        Khi đường linh hay ảnh được chọn thì cho opencv đọc và đưa nó vào canvas
        """
        self.ui.btn_resize.setCurrentIndex(0)
        # Reset toàn bộ data trước khi vẽ lên mới nhé
        self.tool_manager.clear()
        self.tool_manager.reset()

        # Lấy thông tin ảnh xem nào 
        item = self.ui.list_image.currentItem()  # lấy item đang được chọn
        if item is None:
            item = self.ui.list_image.item(0)
            

        self.file_path = item.text()  # đường dẫn ảnh
        self.image = cv2.imread(self.file_path)
        self.get_information_image(self.image)

        if self.image is None:
            print(f"Không thể đọc ảnh: {self.file_path}")
            return

        # Gán ảnh mới vào canvas
        self.canvas.set_image(self.image, self.file_path)  

    def get_information_image(self, image):
        h, w = image.shape[:2]
        self.ui.label_camera.setText(f'IMAGE: {h}x{w}')
    



