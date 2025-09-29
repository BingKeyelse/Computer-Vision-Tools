from libs import*

class ButtonController:
    def __init__(self, ui, tool_manager, canvas):
        """
        Chứa tất cả các hàm liên quan đến Button
        """
        self.ui = ui
        self.tool_manager= tool_manager
        self.canvas= canvas

        # Function ToolBar
        self.link_picutures= []
        self.ui.btn_open.clicked.connect(self.get_link_image)
        # self.ui.list_image.setContextMenuPolicy(Qt.CustomContextMenu) # Không dùng context mặc định mà dùng dạng custom
        # self.ui.list_image.customContextMenuRequested.connect(self.show_list_menu)# Khi bạn bấm chuột phải vào thì phát singal tới slot được định, và auto truyền pos
        
        # Function Tool Shape
        self.ui.btn_shape.currentTextChanged.connect(self.change_tool)
        self.ui.btn_cut.clicked.connect(lambda: (self.tool_manager.cut(), self.canvas.update()))
        self.ui.btn_clear.clicked.connect(lambda: (self.tool_manager.clear(), self.canvas.update()))
        self.ui.btn_undo.clicked.connect(lambda: (self.tool_manager.undo(), self.canvas.update()))
        self.ui.btn_polyundo.clicked.connect(lambda: (self.tool_manager.undo_polygon(), self.canvas.update()))
        self.ui.btn_new.clicked.connect(lambda: (self.tool_manager.reset(), self.canvas.update()))

        
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
        """

        file_path, _ = QFileDialog.getOpenFileName(
            None, # Nếu ở trong MainWindow thì truyền self
            "Chọn ảnh",                # tiêu đề hộp thoại
            "",                        # thư mục mặc định ("" = thư mục hiện tại)
            "Image Files (*.png *.jpg *.jpeg *.bmp)"  # filter chỉ cho phép chọn ảnh
        )
        if file_path:  # Nếu user chọn ảnh (không bấm Cancel)
            # print("Ảnh được chọn:", file_path)
            self.link_picutures.append(file_path)
            # Hiển thị lên QListWidget
            self.ui.list_image.insertItem(0,file_path)  # listWidget là QListWidget trong .ui của bạn

             # Nếu số lượng item > 10 thì xóa item cuối
            if self.ui.list_image.count() > 10:
                self.ui.list_image.takeItem(self.ui.list_image.count() - 1)
        return None
    
    def show_list_menu(self, pos):
        # Lấy widget gọi (list_image)
        widget = self.ui.list_image  

        # Tạo menu context gắn parent là QListWidget
        menu = QMenu(widget)

        # Thêm action chọn ảnh xử lý
        delete_action = QAction("Choose", widget)
        delete_action.triggered.connect(self.choose_selected_item)
        menu.addAction(delete_action)

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
        