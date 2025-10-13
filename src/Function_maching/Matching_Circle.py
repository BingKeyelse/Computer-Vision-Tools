from libs import*

# --- Xoay ảnh quanh tâm ---
def rotate_image_keep_all(img: np.ndarray, 
                          angle: float, 
                          borderValue=(0, 0, 0)):
    """
    ### Xoay ảnh quanh tâm mà không thay đổi kích thước
    - Ảnh được xoay quanh tâm với góc `angle`, giữ nguyên kích thước gốc.
    - Các vùng trống sau khi xoay được tô bằng `borderValue` - màu đen.

    **Parameters**
    ----------
    img : np.ndarray
        Ảnh đầu vào (BGR hoặc grayscale).
    angle : float
        Góc xoay (đơn vị: độ, chiều ngược kim đồng hồ).
    borderValue : tuple[int, int, int], optional
        Màu nền điền vào vùng trống (mặc định: đen).

    **Returns**
    -------
    rotated : np.ndarray
        Ảnh sau khi xoay quanh tâm.
    (h, w) : tuple[int, int]
        Kích thước (cao, rộng) của ảnh đầu ra.
    """
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=borderValue)
    return rotated, (h, w)

class CircleMatcher(BaseMatcher): ## nên blur ảnh để tránh nhiễm viền khi matching
    def __init__(self, temple_path: str, data: list, scale: float) -> None:
        """
        ## Matcher cho hình tròn
        - Input:
            - temple_path: đường dẫn ảnh template
            - data: dữ liệu shape gồm ("box", start, end)
            - scale: tỉ lệ resize ảnh template
        """
        super().__init__(temple_path, data, scale)
        self.template: np.ndarray | None = None

    def load_template(self)-> np.ndarray: # Lấy ở phần cut ảnh ra 
        """
        ## Load và crop ảnh template gốc theo tọa độ. 
        - Bên trong đường tròn là ảnh còn ở ngài thì là màu đen nhé
        - Output:
            - template: ảnh template đã resize theo scale
        """
        _, start, end, angle = self.data
        x1, y1 = map(int, start)
        x2, y2 = map(int, end)

        img = cv2.imread(self.temple_path, cv2.IMREAD_GRAYSCALE)

        # bán kính
        r = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        margin = 5
        extra = 2  # thêm 2px để không sát
        R = r + margin + extra

        # tọa độ vùng crop vuông
        x1_crop, y1_crop = max(0, x1 - R), max(0, y1 - R)
        x2_crop, y2_crop = min(img.shape[1], x1 + R), min(img.shape[0], y1 + R)

        # crop vùng vuông
        cropped = img[y1_crop:y2_crop, x1_crop:x2_crop].copy()
        h, w = cropped.shape[:2]

        # tạo mask hình tròn (center = giữa crop)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w // 2, h // 2), r, 255, -1)

        # giữ phần hình tròn, ngoài vùng là đen
        self.template = cv2.bitwise_and(cropped, cropped, mask=mask)
        self.template = cv2.resize(self.template, (0,0), fx= self.scale, fy= self.scale)
        # cv2.imshow("Result", self.template)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return self.template

    def match(self,
            scene: np.ndarray,
            coarse_scale: float = 0.3,
            coarse_step: int = 10,
            refine_step: int = 2,
            threshold: float = 0.7,
            max_candidates: int = 15,
            max_objects: int = 5,
            pad: int = 20
        ) -> list[dict[str, float | list[int]]]:
        """
        ## Dò tìm template hình hộp chữ nhật trong scene (quét coarse → refine).
        - input:
            - scene: ảnh gốc cần dò tìm
            - coarse_scale: tỉ lệ giảm kích thước ảnh cho bước coarse
            - coarse_step: bước xoay góc trong giai đoạn coarse
            - refine_step: bước xoay tinh trong refine
            - threshold: ngưỡng tương quan tối thiểu
            - max_candidates: số lượng ứng viên tối đa để refine
            - max_objects: số đối tượng giữ lại sau NMS
            - pad: vùng đệm quanh box khi refine
        - output:
            - Danh sách dict chứa:
                - "box": [x1, y1, x2, y2]
                - "angle": góc tìm thấy (độ)
                - "score": độ tương đồng
        """
        
        if self.template is None:
            self.load_template()
        template = self.template

        t0 = time.time()
        print("🌀 [COARSE] scanning...")

        small_scene = cv2.resize(scene, (0, 0), fx=coarse_scale, fy=coarse_scale)
        small_template = cv2.resize(template, (0, 0), fx=coarse_scale, fy=coarse_scale)

        all_boxes, all_scores, all_angles = [], [], []
        angles = np.arange(0, 360, coarse_step)

        # --- Tạo mask tròn từ small_template (1-channel) ---
        h, w = small_template.shape[:2]
        center = (w // 2, h // 2)
        radius = max(1, min(center) - 5)
        mask_base = np.zeros((h, w), dtype=np.uint8)        # <-- 1 channel
        cv2.circle(mask_base, center, radius, 255, -1)

        # 4️⃣ Quét thô các góc
        for angle in angles:
            rotated_t, (tw, th) = rotate_image_keep_all(small_template, angle)
            # cv2.imshow('hhh', rotated_t)
            # cv2.waitKey(0)

            if tw > small_scene.shape[1] or th > small_scene.shape[0]:
                continue

            res = cv2.matchTemplate(small_scene, rotated_t, cv2.TM_CCOEFF_NORMED, mask= mask_base)
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                all_boxes.append([pt[0], pt[1], pt[0] + tw, pt[1] + th])
                all_scores.append(float(res[pt[1], pt[0]]))
                all_angles.append(angle)

        if not all_boxes:
            print("❌ Không có vùng vượt ngưỡng coarse.")
            return []

        # 5️⃣ Lọc ứng viên bằng NMS
        keep = cv2.dnn.NMSBoxes(all_boxes, all_scores, threshold, 0.3)
        keep = sorted(keep.flatten(), key=lambda i: all_scores[i], reverse=True)[:max_candidates]

        coarse_candidates = []
        for i in keep:
            x1, y1, x2, y2 = np.array(all_boxes[i]) / coarse_scale
            coarse_candidates.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "angle": all_angles[i],
                "score": all_scores[i]
            })

        print(f"✅ [COARSE] {len(coarse_candidates)} candidates → refine")

        # --- Tạo mask tròn từ template (1-channel) ---
        h, w = template.shape[:2]
        center = (w // 2, h // 2)
        radius = max(1, min(center) - 5)
        mask_base = np.zeros((h, w), dtype=np.uint8)        # <-- 1 channel
        cv2.circle(mask_base, center, radius, 255, -1)

        # 6️⃣ Quét tinh
        refine_results = []
        for c in coarse_candidates:
            x1, y1, x2, y2 = c["box"]
            angle_c = c["angle"]

            roi = scene[max(0, y1 - 30):min(scene.shape[0], y2 + 30),
                        max(0, x1 - 30):min(scene.shape[1], x2 + 30)]
            if roi.size == 0:
                continue

            best = None
            for a in np.arange(angle_c - 10, angle_c + 10 + 1, refine_step):
                rotated_t, (tw, th) = rotate_image_keep_all(template, a)
                if tw > roi.shape[1] or th > roi.shape[0]:
                    continue

                res = cv2.matchTemplate(roi, rotated_t, cv2.TM_CCOEFF_NORMED, mask= mask_base)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val < threshold:
                    continue

                abs_loc = (max_loc[0] + x1 - 30, max_loc[1] + y1 - 30)
                if (best is None) or (max_val > best["score"]):
                    best = {
                        "shape": "circle",
                        "box": [abs_loc[0], abs_loc[1], abs_loc[0] + tw, abs_loc[1] + th],
                        "angle": a,
                        "score": max_val
                    }

            if best:
                refine_results.append(best)
            
        if not refine_results:
            print("❌ Không có refine result.")
            return []

        boxes = [r["box"] for r in refine_results]
        scores = [r["score"] for r in refine_results]
        keep = cv2.dnn.NMSBoxes(boxes, scores, threshold, 0.3)
        keep = sorted(keep.flatten(), key=lambda i: scores[i], reverse=True)[:max_objects]

        # # 7️⃣ Hiển thị kết quả
        # vis = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
        # results = [refine_results[i] for i in keep]
        # for r in results:
        #     x1, y1, x2, y2 = map(int, r["box"])
        #     cx = int((x1 + x2) / 2)
        #     cy = int((y1 + y2) / 2)
        #     # Vẽ hình tròn
        #     cv2.circle(vis, (cx, cy), radius, (0, 255, 0), 3)
        #     cv2.putText(vis, f"{r['angle']:.1f}° ({r['score']:.2f})",
        #                 (cx - radius, cy - radius - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # print(f"✅ [REFINE] giữ lại {len(keep)} đối tượng.")
        # print(f"⏱ Tổng thời gian: {time.time() - t0:.2f}s")

        return [refine_results[i] for i in keep]


       