from libs import*

class CircleMatcher(BaseMatcher): ## nên blur ảnh để tránh nhiễm viền khi matching
    def __init__(self, image_path, data):
        super().__init__(image_path, data)
        self.template = None

    def load_template(self):
        _, (cx, cy), r = self.data
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        x1, y1 = int(cx - r), int(cy - r)
        x2, y2 = int(cx + r), int(cy + r)
        self.template = img[y1:y2, x1:x2]

        # Optional: Tạo mask tròn để loại vùng nền ngoài viền
        mask = np.zeros_like(self.template, dtype=np.uint8)
        cv2.circle(mask, (r, r), r, 255, -1)
        self.mask = mask

    def match(self, scene, threshold=0.7, max_objects=5):
        if self.template is None:
            self.load_template()

        # Dò tìm trực tiếp bằng matchTemplate với mask
        res = cv2.matchTemplate(scene, self.template, cv2.TM_CCOEFF_NORMED, mask=self.mask)
        loc = np.where(res >= threshold)

        matches = []
        h, w = self.template.shape[:2]

        for pt in zip(*loc[::-1]):
            matches.append({
                "type": "circle",
                "center": (pt[0] + w // 2, pt[1] + h // 2),
                "radius": w // 2,
                "score": float(res[pt[1], pt[0]])
            })

        # Non-Maximum Suppression thủ công đơn giản (tránh trùng tâm)
        matches = sorted(matches, key=lambda m: m["score"], reverse=True)
        final = []
        for m in matches:
            if all(np.linalg.norm(np.array(m["center"]) - np.array(f["center"])) > m["radius"] * 0.8 for f in final):
                final.append(m)
            if len(final) >= max_objects:
                break

        print(f"✅ [CIRCLE] Giữ lại {len(final)} kết quả cuối.")
        return final