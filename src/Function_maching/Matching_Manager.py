from libs import*

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