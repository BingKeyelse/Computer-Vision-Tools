from libs import*

# ================ MatcherFactory ================
class MatcherFactory:
    """
    ## Controller Tool Matching với box tùy biến
    - Chạy lệnh create để tạo kết nối truyền
    - Inputs:
        - data: 'shape' , 'data point' 
    """
    registry = {
        "box": BoxMatcher,
        "circle": CircleMatcher,
        "polygon": PolygonMatcher,
        "oriented_box": OrientedBoxMatcher
    }

    @staticmethod
    def create(mode_data, scale):
        """
        ## Tạo instance cho class xử lý tương ứng với mode
        - input
            - mode_data: dict chứa thông tin để tạo shape matcher
                - "data": list chứa thông tin shape, phần tử đầu là mode ("box" | "circle" | ...)
                - "link": đường dẫn ảnh hoặc metadata cần thiết cho class
            - scale: tỉ lệ scale của ảnh (ví dụ 1.0 = 100%)
        - output
            - Instance của class tương ứng (ví dụ BoxMatcher, CircleMatcher, ...)
        - Lưu ý:
            - Nếu mode không được đăng ký, sẽ raise lỗi ValueError
        """
        
        mode = mode_data["data"][0]   # "box" | "circle" | "polygon" | "oriented"
        cls = MatcherFactory.registry.get(mode)
        if cls:
            return cls(mode_data["link"], mode_data["data"], scale)
        raise ValueError(f"Không hỗ trợ mode {mode}")

## How to use them
# main.py
# import cv2
# from core.matcher_factory import MatcherFactory

# data_list = [
#     {'mode': 0, 'link': 'images/image.jpg', 'data': ('box', (1451.86, 2013.24), (2429.44, 3619.97))},
#     {'mode': 0, 'link': 'images/image.jpg', 'data': ('circle', (500, 500), 80)}
# ]

# scene = cv2.imread("images/scene.jpg", cv2.IMREAD_GRAYSCALE)
# results = []

# for obj in data_list:
#     matcher = MatcherFactory.create(obj)
#     matcher.load_template()
#     res = matcher.match(scene)
#     results.append(res)

# print(results)