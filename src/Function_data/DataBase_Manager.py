from libs import*

class DatabaseController:
    def __init__(self, ui, tool_manager, canvas, canvas_Sample, cam_function):
        """
        Chứa tất cả các hàm liên quan đến Button
        """
        self.ui = ui
        self.tool_manager= tool_manager
        self.canvas_Image= canvas
        self.canvas_Sample= canvas_Sample

        self.camera_function = cam_function