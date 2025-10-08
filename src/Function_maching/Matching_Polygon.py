from libs import*

class PolygonMatcher(BaseMatcher):
    def load_template(self):
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.template_contour = max(contours, key=cv2.contourArea)

    def match(self, scene):
        _, thresh = cv2.threshold(scene, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = min(contours, key=lambda c: cv2.matchShapes(self.template_contour, c, 1, 0.0))
        return {"type": "polygon", "contour": best}