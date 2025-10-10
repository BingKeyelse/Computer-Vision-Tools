from libs import*

class ButtonController:
    def __init__(self, ui, tool_manager, canvas, canvas_Sample, cam_function, data_functions, Matching_Controller):
        """
        Chứa tất cả các hàm liên quan đến Button
        """
        self.ui = ui
        self.tool_manager= tool_manager
        self.canvas_Image= canvas
        self.canvas_Sample= canvas_Sample

        self.camera_function = cam_function
        self.Data_Functions= data_functions
        self.Matching_Controller = Matching_Controller

        # Function Camera
        self.image=None
        self.file_path=None
        self.scale= 1
        self.image_resized= None
        self.ui.btn_resize.currentTextChanged.connect(self.resize_image) 

        # Function ToolBar
        self.link_picutures= []
        self.ui.btn_open.clicked.connect(self.get_link_image)
        self.ui.btn_save.clicked.connect(self.save_current_frame) #Ken

        # self.ui.list_image.setContextMenuPolicy(Qt.CustomContextMenu) # Không dùng context mặc định mà dùng dạng custom
        # self.ui.list_image.customContextMenuRequested.connect(self.show_list_menu)# Khi bạn bấm chuột phải vào thì phát singal tới slot được định, và auto truyền pos
        
        # Function Tool Shape
        self.data_SHAPE=[]

        # Tạo Sample_button, truyền chính self (instance ButtonController)
        self.sample_button = Sample_button(self)

        # Tạo Matching button, truyền chính self ( instance ButtonController)
        self.matching_button = Matching_button(self)

        self.ui.btn_shape.currentTextChanged.connect(self.change_tool) # Singal tự gửi được Toolname của QListWidget
        self.ui.btn_cut.clicked.connect(lambda: (self.get_shape_and_update(), self.canvas_Image.update()))
        self.ui.btn_clear.clicked.connect(lambda: (self.tool_manager.clear(), self.canvas_Image.update()))
        self.ui.btn_clear.clicked.connect(lambda: (self.tool_manager.reset(), self.canvas_Image.update()))
        self.ui.btn_undo.clicked.connect(lambda: (self.tool_manager.undo(), self.canvas_Image.update()))
        self.ui.btn_undo.clicked.connect(lambda: (self.tool_manager.reset(), self.canvas_Image.update()))
        self.ui.btn_polyundo.clicked.connect(lambda: (self.tool_manager.undo_polygon(), self.canvas_Image.update()))
        self.ui.btn_new.clicked.connect(lambda: (self.tool_manager.reset(), self.canvas_Image.update()))

    def get_shape_and_update(self):
        """
        ## Cập nhập data shape và update Shape bar
        - Truyền vào biến rồi lấy lại biến đó 
        - Để biết nó lấy shape được cut và update phần phím Sample 
        """
        if len(self.data_SHAPE)<12:
            self.data_SHAPE= self.tool_manager.cut(self.data_SHAPE)
            print("=============================")
            print(len(self.data_SHAPE))
            print(self.data_SHAPE)
            self.sample_button.show_Sample()

    def resize_image(self, size_text)-> None:
        """ 
        ## Dùng để lựa chọn để tùy chỉnh kích thước của ảnh 
        - Resize và lấy thông tin ảnh
        - Gửi vào canvas.set_image với ảnh resize, link và tỉ lệ resized
        """

        # Reset toàn bộ data trước khi vẽ lên mới nhé
        # self.tool_manager.clear()
        # self.tool_manager.reset()

        self.scale= int(size_text)/100

        if self.image is None:
            return None
    
        h, w = self.image.shape[:2]
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)

        if new_w <= 0 or new_h <= 0:
            return None  # Tránh resize về 0

        # Resize ảnh bằng OpenCV
        resized = cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Lưu lại ảnh đã resize (nếu bạn muốn giữ)
        self.image_resized = resized  
        self.get_information_image(self.image_resized)

        # Nếu bạn đang có canvas để hiển thị thì update luôn
        if hasattr(self, "canvas"):
            self.canvas_Image.set_image(resized, self.file_path, self.scale)
        
    def change_tool(self, tool_name)-> None:
        """
        ### Singal tự gửi được Toolname của QListWidget
        ## Hàm set cho ToolManager xem sài hàm nào
        - BoxTool
        - CircleTool
        - PolygonTool
        - OrientedBoxTool
        - input
            - tool_name: tự nhận diện được
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

    def get_link_image(self)-> None:
        """
        ## Mở hộp thoại chọn file ảnh.
        - Nếu chọn thì ảnh sẽ vào ListWidget theo dạng insertItem
        - Xong show ảnh luôn
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

    def save_current_frame(self):
        """
        Lưu ảnh hiện tại từ camera realtime xuống file
        """
        if not self.active_cam:
            QMessageBox.information(self, "Info", "Không có camera đang chạy.")
            return

        frame = self.active_cam.get_frame()
        if frame is None:
            QMessageBox.warning(self, "Warning", "Không lấy được frame từ camera!")
            return

        # Mở hộp thoại lưu file
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu ảnh",
            "",  # thư mục mặc định
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            # Lưu ảnh bằng OpenCV (ảnh đang là BGR)
            cv2.imwrite(file_path, frame)
            QMessageBox.information(self, "Success", f"Đã lưu ảnh tại:\n{file_path}")

    
    def show_list_menu(self, pos):
        """
        ## Tạo list action tương tác chuột phải với ListWidget
        """
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
        """
        ## Copy link ảnh với action - ListWidget
        """
        item = self.ui.list_image.currentItem()
        if item:  # tránh lỗi nếu chưa chọn gì
            QApplication.clipboard().setText(item.text())

    def delete_selected_item(self):
        """
        ## Xóa ảnh đối tượng đang hiện thị với action - ListWidget
        """
        item = self.ui.list_image.currentItem()
        if item:
            row = self.ui.list_image.row(item)
            self.ui.list_image.takeItem(row)
    
    def choose_selected_item(self):
        """
        ## Chọn ảnh để hiển thị với action - ListWidget 
        - Đưa phần resize về 100%, clear toàn bộ data mà Tool Manager đang vẽ và đang được cut ra rồi
        - Khi đường link hay ảnh được chọn thì cho opencv đọc và đưa nó vào canvas cập nhập giao diện
        - Truy cập vào hàm self.canvas_Image.set_image truyền 2 đối số (ảnh resize, link gốc)
        để cập nhập ảnh vào canvas
        """
        self.ui.btn_resize.setCurrentIndex(0)
        # Reset toàn bộ data trước khi vẽ lên mới nhé
        self.tool_manager.clear()
        self.tool_manager.reset()

        # Lấy thông tin ảnh xem nào 
        item = self.ui.list_image.currentItem()  # lấy item đang được chọn
        if item is None:
            item = self.ui.list_image.item(0)
        
        if item is None:
            return
            
        self.file_path = item.text()  # đường dẫn ảnh
        self.image = cv2.imread(self.file_path)
        self.image_resized = self.image
        self.get_information_image(self.image)

        if self.image is None:
            # print(f"Không thể đọc ảnh: {self.file_path}")
            return
        self.ui.stackwidget.setCurrentWidget(self.ui.page_sub_1)

        # Gán ảnh mới vào canvas
        self.canvas_Image.set_image(self.image, self.file_path)  

    def get_information_image(self, image):
        """## Dùng để lấy thông tin kích thước của ảnh"""
        h, w = image.shape[:2]
        self.ui.label_camera.setText(f'IMAGE: {h}x{w}')

    
    def check_value_int_line_edit(self, obj)-> int:
        """
        ## Check nếu giá trị không đúng trả về giá trị 0
        - Inputs:
            - obj: truyền vào label hay editText muốn lấy giá trị
        """
        value= obj.text()
        if self.Check_convert_str_to_int(value)==True:
            return int(value)
        else:
            return 0
        
    
    def Check_convert_str_to_int(self,number_check)-> bool:
        """
        ## Kiểm tra xem có chuyển được string sang INT không
        - Inputs:
            - number_check: giá trị cần kiểm
        """
        try:
            int(number_check)
            return True
        except ValueError:
            return False
    

class Sample_button:
    """
    ## Chức năng button dành riêng cho phần Sample
    """
    def __init__(self, controller: ButtonController):
        """
        ## Khởi tạo biến để chạy chức năng cho phần Sample Button
        - Phải có đủ các đối tượng thừa kết từ ButtonController
            - ui: phần giao diện thừa kế
            - tool_manager: phần thừa thế Tool Manager 
            - canvas: phần thừa kế để hiện thị ảnh ở phần chính giữa
            - canvas_Sample: phần thừa kế để hiện thị ảnh Sample
            - data_SHAPE: dữ liệu data_SHAPE kế thừa từ ButtonController để tiện sử dụng và cập nhập 
            - Data_Functions: thừa kế class DatabaseController
        """
        # super().__init__(ui, tool_manager, canvas, canvas_Sample) phải lấy lại hết đối số này nhé
        self.controller= controller
        self.ui= controller.ui
        self.tool_manager = controller.tool_manager
        self.canvas_Image = controller.canvas_Image
        self.canvas_Sample = controller.canvas_Sample
        # self.data_SHAPE = self.controller.data_SHAPE
        self.Data_Functions = controller.Data_Functions

        # Setup click with stick Sample
        for i in range(1, 12):
            btn_stick = getattr(self.ui, f"btn_stick_{i}", None)
            if btn_stick:
                btn_stick.clicked.connect(lambda _, idx=i: self.stick_Sample(idx))

        # Setup click with sample Sample
        for i in range(1, 12):
            btn_sample = getattr(self.ui, f"btn_sample_{i}", None)
            if btn_sample:
                btn_sample.clicked.connect(lambda _, idx=i: self.show_image_Sample(idx))
             

        # Setup click with delete Sample
        for i in range(1, 12):
            btn_delete = getattr(self.ui, f"btn_delete_{i}", None)
            if btn_delete:
                btn_delete.clicked.connect(lambda _, idx=i: self.remove_Sample(idx))
    
    def stick_Sample(self, idx=None)-> None:
        """
        ## Bấm stick nào được chọn và thay đổi mode
        - input
            - idx: giá trị idx muốn thay đổi mode
        """
        if idx is None:
            return
        # Lấy giá trị mode phù hợp với idx
        mode = self.controller.data_SHAPE[idx-1]['mode']
        # Toggle mode
        self.controller.data_SHAPE[idx-1]['mode'] = 1 if mode == 0 else 0
        self.control_stick_Sample()
    
    def control_stick_Sample(self):
        """
        ## Bộ kiểm soát stick hiển thị chọn hay không được chọn
        """
        for idx, data in enumerate(self.controller.data_SHAPE):
            btn_stick = getattr(self.ui, f"btn_stick_{idx+1}", None)
            if not btn_stick:
                continue

            if data['mode'] == 1:
                btn_stick.setStyleSheet("background-color: green")
            else:
                btn_stick.setStyleSheet("background-color: white")

    def remove_Sample(self,idx=None)-> None:
        """
        ## Xóa Sample vởi chỉ định rõ ràng idx
        - input
            - idx: thứ tự xóa theo nút nhấn trên Sample bar
        """
        if idx is not None:
            self.controller.data_SHAPE.pop(idx-1)
            # self.tool_manager.remove_SHAPE(idx-1)
            self.control_stick_Sample()
            self.show_Sample()
            self.canvas_Image.update()
    
    def undo_Sample(self)-> None:
        """
        ## Lệnh này tương tác undo với nút undo bên Tool Shape
        """
        if len(self.controller.data_SHAPE)>0:
            self.controller.data_SHAPE.pop()
            self.show_Sample()
            self.canvas_Image.update()
    
    def clear_Sample(self)-> None:
        """
        ## Lệnh này là để tương tác với clear bên Tool Shape
        """
        
        self.controller.data_SHAPE=[]
        self.show_Sample()
        self.canvas_Image.update()
            
    def show_Sample(self)-> None:
        """
        ## Hiện thị nút nhấn bên Sample Tool
        """
        leng= len(self.controller.data_SHAPE)
        self.hide_Sample() # Reset Sample Tool

        for i in range(1,leng+1):  # từ 1 -> 11
            btn_sample = getattr(self.ui, f"btn_sample_{i}", None)
            btn_stick = getattr(self.ui, f"btn_stick_{i}", None)
            btn_delete = getattr(self.ui, f"btn_delete_{i}", None)

            if btn_sample:
                btn_sample.setVisible(True)   # hoặc btn_sample.hide()
            if btn_stick:
                btn_stick.setVisible(True)   # hoặc btn_stick.hide()
            if btn_delete:
                btn_delete.show()
                
        self.control_stick_Sample()
    
    def hide_Sample(self)-> None:
        """
        ## Ẩn toàn bộ nút nhấn bên Sample
        - Thường dùng để reset
        """
        for i in range(1, 12):  # từ 1 -> 11
            btn_sample = getattr(self.ui, f"btn_sample_{i}", None)
            btn_stick = getattr(self.ui, f"btn_stick_{i}", None)
            btn_delete = getattr(self.ui, f"btn_delete_{i}", None)

            if btn_sample:
                btn_sample.setVisible(False)   # hoặc btn_sample.hide()
            if btn_stick:
                btn_stick.setVisible(False)   # hoặc btn_stick.hide()
            if btn_delete:
                btn_delete.hide()
    
    def show_image_Sample(self, idx)-> None:
        """
        ## Show ảnh trên canvas Sample để hiển thị ảnh
        - Đọc link ảnh cho vào crop shape để lấy định dạng 100%
        - Ảnh sẽ được đưa cho canvas Sample để nó hiện thị  
        """
        # Show StackWidget
        self.ui.stackedWidget_processing.setCurrentWidget(self.ui.page_detail_sample)
        link = self.controller.data_SHAPE[idx-1]['link']
        shape = self.controller.data_SHAPE[idx-1]['data']

        # Lấy ảnh crop từ tool_manager
        image = self.tool_manager.crop_shape(link, shape)

        if image is None:
            print(f"[ERROR] Không load được ảnh từ {link}")
            return

        # Truyền cv2 image vào canvas
        self.canvas_Sample.set_image(image, link)

def rotate_image_keep_all(img, angle, borderValue=(255, 255, 255)):
    """Rotate image around center but keep full content (expanded canvas)."""
    (h, w) = img.shape[:2]
    angle_rad = np.deg2rad(angle)
    cos, sin = abs(np.cos(angle_rad)), abs(np.sin(angle_rad))
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return M, (new_w, new_h)

class Matching_button:
    """
    ## Chức năng button dành riêng cho phần Matching Controller
    """
    def __init__(self, controller: ButtonController):
        """
        ## Khởi tạo biến để chạy chức năng cho phần Sample Button
        - Phải có đủ các đối tượng thừa kết từ ButtonController
            - ui: phần giao diện thừa kế
            - tool_manager: phần thừa thế Tool Manager 
            - canvas: phần thừa kế để hiện thị ảnh ở phần chính giữa
            - canvas_Sample: phần thừa kế để hiện thị ảnh Sample
            - data_SHAPE: dữ liệu data_SHAPE kế thừa từ ButtonController để tiện sử dụng và cập nhập 
            - Data_Functions: thừa kế class DatabaseController
        """
        # super().__init__(ui, tool_manager, canvas, canvas_Sample) phải lấy lại hết đối số này nhé
        self.controller= controller
        self.ui= controller.ui
        self.tool_manager = controller.tool_manager
        self.canvas_Image = controller.canvas_Image
        self.canvas_Sample = controller.canvas_Sample
        # self.data_SHAPE = controller.data_SHAPE
        # self.scale = controller.scale
        # self.image_resized = controller.image_resized
        self.Data_Functions = controller.Data_Functions
        self.Matching_Controller = controller.Matching_Controller

        self.ui.btn_matching_process.clicked.connect(lambda: self.ui.stackedWidget_processing.setCurrentWidget(self.ui.page_matching))
        self.ui.btn_run_matching.clicked.connect(self.matching_processing)

        self.data_matching=[]
        # Lấy giá trị trong dataset
        self.get_data_from_Database()

        # Define thanh kéo
        self.define_UI()

        
        
    def define_UI(self):
        """
        ## Liên kết UI với các sự kiện điều khiển
        - Description:
            Gắn các sự kiện (signal) của slider và line edit với hàm `adjust_value`
            để khi người dùng thay đổi giá trị trên giao diện, dữ liệu được cập nhật tự động.
        - Connect:
            - slider_coarse_scale_matching → adjust_value()
            - slider_limit_score_matching → adjust_value()
            - edit_max_objects_matching (khi người dùng nhấn Enter hoặc rời khỏi ô) → adjust_value()
        """
        self.ui.slider_coarse_scale_matching.valueChanged.connect(self.adjust_value)
        self.ui.slider_limit_score_matching.valueChanged.connect(self.adjust_value)
        self.ui.edit_max_objects_matching.editingFinished.connect(self.adjust_value)
    
    def adjust_value(self):
        """
        ## Cập nhật dữ liệu khi người dùng thay đổi giá trị trên UI
        - Description:
            Lấy giá trị hiện tại từ các slider và ô nhập, 
            lưu vào `self.data_matching`, sau đó:
            1️⃣ Cập nhật lại hiển thị UI (`update_UI_matching()`)
            2️⃣ Ghi dữ liệu mới vào database (`update_data_matching_values`)
        - Process:
            - coarse_scale và limit_score được nhân / chia 100 để chuẩn hóa về dạng 0.x
            - max_object được lấy từ ô nhập bằng `check_value_int_line_edit`
        """
        self.data_matching[0]=int(self.ui.slider_coarse_scale_matching.value())/100.0
        self.data_matching[1]=int(self.ui.slider_limit_score_matching.value())/100.0
        self.data_matching[2]=self.controller.check_value_int_line_edit(self.ui.edit_max_objects_matching)

        self.update_UI_matching()

        self.Data_Functions.update_data_matching_values(0, "Data_matching", self.data_matching)
        
    def get_data_from_Database(self):
        """
        ## Cập nhật dữ liệu khi người dùng thay đổi giá trị trên UI
        - Description:
            Lấy giá trị hiện tại từ các slider và ô nhập, 
            lưu vào `self.data_matching`, sau đó:
            1️⃣ Cập nhật lại hiển thị UI (`update_UI_matching()`)
            2️⃣ Ghi dữ liệu mới vào database (`update_data_matching_values`)
        - Process:
            - coarse_scale và limit_score được nhân / chia 100 để chuẩn hóa về dạng 0.x
            - max_object được lấy từ ô nhập bằng `check_value_int_line_edit`
        """
        # Lấy giá trị
        data_getting= self.Data_Functions.get_data_matching_values(0,"Data_matching")
        print(data_getting)

        for k, v in data_getting[0].items():
            if k != 'id':
                self.data_matching.append(v)
        print(self.data_matching)
        
        self.update_UI_matching()

    def update_UI_matching(self):
        """
        ## Hiển thị giá trị hiện tại lên UI
        - Description:
            Cập nhật các label, slider và ô nhập (line edit) theo dữ liệu trong `self.data_matching`.
            Dùng nhân 100 để chuyển giá trị 0.x sang dạng phần trăm cho dễ đọc.
        - UI Updated:
            - label_coarse_scale_matching
            - slider_coarse_scale_matching
            - label_limit_score_matching
            - slider_limit_score_matching
            - edit_max_objects_matching
        """
        self.ui.label_coarse_scale_matching.setText(f'{int(self.data_matching[0]*100)}')
        self.ui.slider_coarse_scale_matching.setValue(int(self.data_matching[0]*100))

        self.ui.label_limit_score_matching.setText(f'{int(self.data_matching[1]*100)}')
        self.ui.slider_limit_score_matching.setValue(int(self.data_matching[1]*100))

        self.ui.edit_max_objects_matching.setText(f'{self.data_matching[2]}')

    def matching_processing(self):
        mode_NMS=1
        if self.controller.image_resized is not None:
            gray = cv2.cvtColor(self.controller.image_resized, cv2.COLOR_BGR2GRAY)
            draw_img = self.controller.image_resized.copy()
            res_data=[]
            template_list=[]

            for data in self.controller.data_SHAPE:
                if data['mode'] ==1: # Chỉ chấp nhận mode 1 mới cho kích hoạt chạy
                    matcher = self.Matching_Controller.create(data, self.controller.scale)
                    template= matcher.load_template()
                    res = matcher.match(scene = gray, 
                                        coarse_scale= self.data_matching[0],
                                        threshold= self.data_matching[1],
                                        max_objects= self.data_matching[2])
                    res_data.extend([
                                    {
                                        **r,
                                        "template_shape": template.shape[:2],
                                        "template_angle": data.get("angle", 0),
                                        "template_name": data.get("name", "unknown")
                                    }
                                    for r in res
                                ])
                    
                    if mode_NMS==0:
                        print('Run not NMS')
                        if res:
                            # draw_img = self.controller.image_resized.copy()
                            gray = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)

                            # Lấy template thực tế từ matcher để biết kích thước
                            template = matcher.template
                            h_t, w_t = template.shape[:2]  # Kích thước template

                            for r in res:
                                x1, y1, x2, y2 = r["box"]
                                angle = r["angle"]
                                score = r["score"]

                                # --- Tính lại vị trí polygon thật của template ---
                                M_rot, (new_w, new_h) = rotate_image_keep_all(template, angle)

                                # 4 góc template gốc
                                corners_t = np.array([
                                    [0, 0],
                                    [w_t, 0],
                                    [w_t, h_t],
                                    [0, h_t]
                                ], dtype=np.float32)

                                ones = np.ones((4, 1), dtype=np.float32)
                                corners_h = np.hstack([corners_t, ones])
                                rotated_t = (M_rot @ corners_h.T).T

                                # Offset dịch polygon về vị trí match
                                offset_x = rotated_t[:, 0].min()
                                offset_y = rotated_t[:, 1].min()
                                rotated_in_scene = rotated_t - [offset_x, offset_y] + [x1, y1]
                                rotated_in_scene = rotated_in_scene.astype(np.int32)

                                # --- Vẽ polygon xoay ---
                                cv2.polylines(draw_img, [rotated_in_scene], isClosed=True, color=(0, 255, 0), thickness=2)

                                # Tâm trung bình để ghi chữ
                                cx, cy = np.mean(rotated_in_scene, axis=0).astype(int)
                                cv2.putText(draw_img, f"angle: {angle:.1f}° | score: {score:.2f}",
                                            (int(cx), int(cy) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            print(res_data)
            if mode_NMS == 1:
                print('Run NMS')

                # --- 1. Kiểm tra dữ liệu ---
                if len(res_data) == 0:
                    print("❌ Không có kết quả nào từ tất cả matcher.")
                    return

                # --- 2. Chuẩn bị dữ liệu cho NMS ---
                boxes_xywh = []
                for r in res_data:
                    x1, y1, x2, y2 = r["box"]
                    boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

                scores = [r["score"] for r in res_data]

                # --- 3. Thực hiện NMS ---
                keep = cv2.dnn.NMSBoxes(
                    bboxes=boxes_xywh,
                    scores=scores,
                    score_threshold=self.data_matching[1],  # threshold
                    nms_threshold=0.3                       # mức chồng lấn cho phép
                )

                if len(keep) == 0:
                    print("❌ Không còn box nào sau NMS.")
                    return

                # --- 4. Lấy danh sách giữ lại ---
                keep = keep.flatten()
                filtered_results = [res_data[i] for i in keep]

                print(f"✅ Giữ lại {len(filtered_results)} box sau NMS")

                # --- 5. Vẽ kết quả ---
                for r in filtered_results:
                    x1, y1, x2, y2 = r["box"]
                    angle = r["angle"]
                    score = r["score"]

                    # Lấy kích thước template gốc
                    h_t, w_t = r["template_shape"]
                    name = r.get("template_name", "unknown")

                    # Xoay template ảo để dựng polygon đúng góc
                    dummy_template = np.zeros((h_t, w_t), dtype=np.uint8)
                    M_rot, (new_w, new_h) = rotate_image_keep_all(dummy_template, angle)

                    corners_t = np.array([
                        [0, 0],
                        [w_t, 0],
                        [w_t, h_t],
                        [0, h_t]
                    ], dtype=np.float32)
                    ones = np.ones((4, 1), dtype=np.float32)
                    corners_h = np.hstack([corners_t, ones])
                    rotated_t = (M_rot @ corners_h.T).T

                    offset_x = rotated_t[:, 0].min()
                    offset_y = rotated_t[:, 1].min()
                    rotated_in_scene = rotated_t - [offset_x, offset_y] + [x1, y1]
                    rotated_in_scene = rotated_in_scene.astype(np.int32)

                    # Vẽ polygon xoay
                    cv2.polylines(draw_img, [rotated_in_scene], isClosed=True, color=(0, 255, 0), thickness=2)
                    cx, cy = np.mean(rotated_in_scene, axis=0).astype(int)
                    cv2.putText(draw_img, f"{name} | {score:.2f} | {angle:.1f}°",
                                (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # --- 4. Hiển thị ---
            cv2.imshow("Filtered Matching", draw_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else: 
            print('none data')
            print(self.controller.scale)
                


        
        
