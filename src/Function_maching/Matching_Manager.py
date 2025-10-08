from libs import*

class MatcherFactory:
    registry = {
        "box": BoxMatcher,
        "circle": CircleMatcher,
        "polygon": PolygonMatcher
        # "oriented": OrientedBoxMatcher
    }

    @staticmethod
    def create(mode_data):
        mode = mode_data["data"][0]   # "box" | "circle" | "polygon" | "oriented"
        cls = MatcherFactory.registry.get(mode)
        if cls:
            return cls(mode_data["link"], mode_data["data"])
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