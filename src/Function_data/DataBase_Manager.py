from libs import*

class DatabaseController:
    def __init__(self, ui, tool_manager, canvas, canvas_Sample, cam_function, current_file_path):
        """
        Quản lý và thao tác với database ứng dụng.
        """
        

        self.ui = ui
        self.tool_manager= tool_manager
        self.canvas_Image= canvas
        self.canvas_Sample= canvas_Sample

        self.camera_function = cam_function
        self.current_file_path= current_file_path

        # Lấy đường dẫn
        # Lấy đường dẫn tuyệt đối của file đang chạy 
        
        self.database=['Data_matching.db']
        self.db_connections={}
        self.check_or_create_database()
    
    def check_or_create_database(self):
        """
        Tạo database và bảng mặc định nếu chưa tồn tại.
        """
        for db_name in self.database:
            print(self.current_file_path)
            db_path = os.path.join(self.current_file_path, "data", db_name)
            # print("Database Path:", db_path)  # Kiểm tra đường dẫn
         
            # Tạo connect tới database
            conn = sqlite3.connect(db_path)

            # Tạo trỏ chuột với database
            cursor = conn.cursor()
            
            # Tạo bảng nếu chưa tồn tại nếu có rồi thì bỏ qua
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Data_matching (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coarse_scale FLOAT ,
                    limit_score FLOAT ,
                    max_object INT 
                )
            ''')
            conn.commit() # Commit kết nối
        
            # 🔹 Kiểm tra bảng có dữ liệu chưa
            cursor.execute("SELECT COUNT(*) FROM Data_matching")
            count = cursor.fetchone()[0]

            # 🔹 Nếu chưa có, thêm dòng mặc định
            if count == 0:
                cursor.execute('''
                    INSERT INTO Data_matching (coarse_scale, limit_score, max_object)
                    VALUES (0.6, 0.7, 1)
                ''')
                conn.commit()
        
            # Lưu kết nối
            self.db_connections[db_name] = conn
    
    def get_data_matching_values(self, idx, table_name)-> list:
        """
        Lấy toàn bộ dữ liệu từ bảng được chỉ định.

        Args:
            idx (int):  Chỉ số (0-based) trong self.database để chọn file DB.
            table_name (str): Tên bảng trong database (ví dụ "Data_matching").
        """

        # Lấy tên database theo idx
        db_name = self.database[idx]
        conn = self.db_connections.get(db_name)
        if not conn:
            raise ValueError(f"Database '{db_name}' chưa được kết nối")
    
        # Lấy trỏ chuột ra 
        cursor = conn.cursor()

        # Truy vấn tất cả dữ liệu trong bảng
        try:
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()

            # Lấy tên các cột để biết cấu trúc bảng
            col_names = [description[0] for description in cursor.description]

            # Ghép lại thành danh sách dict để dễ dùng
            results = [dict(zip(col_names, row)) for row in rows]
            # print(results)
            return results

        except sqlite3.OperationalError as e:
            print(f"Lỗi khi truy cập bảng '{table_name}': {e}")
            return None
    
    def update_data_matching_values(self, idx, table_name, new_values):
        """
        Cập nhật lại 3 giá trị (coarse_scale, limit_score, max_object) trong bảng chỉ định.
        - idx: chỉ số database trong self.database
        - table_name: tên bảng (ví dụ "Data_matching")
        - new_values: list hoặc tuple gồm 3 giá trị [coarse_scale, limit_score, max_object]
        """
        
        # Lấy tên database theo idx
        db_name = self.database[idx]
        conn = self.db_connections.get(db_name)
        if not conn:
            raise ValueError(f"Database '{db_name}' chưa được kết nối")
        
        # Lấy trỏ chuột ra 
        cursor = conn.cursor()

        # 🔹 Cập nhật vào dòng đầu tiên (id = 1), hoặc tùy logic bạn muốn
        cursor.execute(f'''
            UPDATE {table_name}
            SET coarse_scale = ?,
                limit_score = ?,
                max_object = ?
            WHERE id = 1
        ''', (new_values[0], new_values[1], new_values[2]))

        conn.commit()
