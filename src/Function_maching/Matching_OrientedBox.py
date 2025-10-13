from libs import*

def rotate_image_keep_all(img: np.ndarray, angle: float, borderValue: tuple[int, int, int] = (255, 255, 255)
    ) -> tuple[np.ndarray, tuple[int, int]]:
    """
    ## Xoay ảnh quanh tâm nhưng vẫn giữ toàn bộ nội dung (canvas mở rộng)
    - input
        - img: ảnh đầu vào (numpy.ndarray)
        - angle: góc xoay (độ, chiều ngược kim đồng hồ)
        - borderValue: màu nền viền khi mở rộng canvas (mặc định trắng)
    - output
        - M: ma trận xoay 2x3 để sử dụng trong `cv2.warpAffine`
        - (new_w, new_h): kích thước mới của ảnh sau khi xoay
    - Ghi chú
        - Hàm này chỉ tính toán ma trận và kích thước, không thực hiện xoay.
        - Dùng khi cần xoay ảnh mà không bị mất phần nào của nội dung.
    """
    (h, w) = img.shape[:2]
    angle_rad = np.deg2rad(angle)
    cos, sin = abs(np.cos(angle_rad)), abs(np.sin(angle_rad))
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return M, (new_w, new_h)

def extract_oriented_object(img: np.ndarray, start: tuple, end: tuple, angle_rad:float
    )-> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ## Chuẩn hóa vùng oriented box (nghiêng) về hướng nằm ngang
    - input
        - img: ảnh đầu vào (numpy.ndarray)
        - start: tọa độ điểm đầu (x1, y1)
        - end: tọa độ điểm đối diện đường chéo (x2, y2)
        - angle_rad: góc nghiêng của box (radian)
    - output
        - cropped: vùng đối tượng đã được xoay thẳng và crop ra
        - rotated: ảnh toàn cảnh sau khi xoay toàn bộ để “dựng thẳng”
        - box_rotated: tọa độ polygon 4 đỉnh của box trong ảnh đã xoay
    - Giải thích
        - Hàm này dùng để xử lý template hoặc đối tượng nghiêng (oriented box)
          và đưa nó về hướng chuẩn (nằm ngang) để dễ matching.
    """
    # Chuyển góc sang độ
    angle_deg = np.degrees(angle_rad)

    # Tâm và kích thước box
    cx, cy = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    w = abs(end[0] - start[0])
    h = abs(end[1] - start[1])

    # Tạo rotated rect (giống cv2.minAreaRect)
    rect = ((cx, cy), (w, h), angle_deg)

    # Lấy 4 điểm polygon từ rect (theo hướng nghiêng)
    box = cv2.boxPoints(rect).astype(np.float32)

    # --- Tạo ma trận xoay để "dựng thẳng" box ---
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Tính kích thước mới sau khi xoay toàn ảnh
    h_img, w_img = img.shape[:2]
    new_w = int(h_img * sin + w_img * cos)
    new_h = int(h_img * cos + w_img * sin)

    # Cập nhật offset để không bị crop ngoài
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    # Xoay toàn ảnh để đối tượng về thẳng
    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    # Chuyển box điểm sang không gian mới (sau khi xoay)
    ones = np.ones((4, 1), dtype=np.float32)
    box_h = np.hstack([box, ones])
    box_rotated = (M @ box_h.T).T

    # Crop vùng bounding box mới
    x_min, y_min = box_rotated[:, 0].min(), box_rotated[:, 1].min()
    x_max, y_max = box_rotated[:, 0].max(), box_rotated[:, 1].max()

    cropped = rotated[int(y_min):int(y_max), int(x_min):int(x_max)]
    return cropped, rotated, box_rotated.astype(np.int32)


# ================ OrientedBoxMatcher ================
class OrientedBoxMatcher(BaseMatcher):  
    """
    ## Lớp xử lý template matching cho vùng oriented box (nghiêng).
    - Tự động dựng thẳng template và quét đa góc (coarse → refine).
    """
    def __init__(self, temple_path: str, data: list, scale: float):
        """
        - input:
            - temple_path: đường dẫn đến ảnh template
            - data: thông tin vùng oriented box (x1, y1, x2, y2, angle)
            - scale: tỉ lệ phóng template
        """
        super().__init__(temple_path, data, scale)
        self.template: np.ndarray | None = None

    def load_template(self)-> np.ndarray:
        """
        ## Load và dựng thẳng vùng oriented box trong template.
        - output:
            - template đã được dựng thẳng, resize theo scale
        """
        _, (x1, y1), (x2, y2), angle = self.data
        img = cv2.imread(self.temple_path, cv2.IMREAD_GRAYSCALE)

        self.template, rotated, box_rotated = extract_oriented_object(img, (x1, y1), (x2, y2), angle)

        ## Test
        # Vẽ vùng oriented box sau xoay
        # Vẽ vùng oriented box sau xoay
        # vis = rotated.copy()
        # cv2.polylines(vis, [box_rotated], isClosed=True, color=(0, 255, 0), thickness=2)
        # vis= cv2.resize(vis, (0,0), fx= 0.5, fy=0.5)
        # cv2.imshow("Rotated Scene", vis)
        # cv2.imshow("Aligned Crop", self.template)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        self.template = cv2.resize(self.template, (0,0), fx= self.scale, fy= self.scale)
        return self.template

    def match(
        self,
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
        
        # small_scene = cv2.GaussianBlur(small_scene, (3, 3), 0)
        # small_template = cv2.GaussianBlur(small_template, (3, 3), 0)

        # 2. Làm mượt để giảm vùng biên trắng
        small_scene = cv2.bilateralFilter(small_scene, 5, 50, 50)
        small_template = cv2.bilateralFilter(small_template, 5, 50, 50)

        # 3. Cân sáng
        small_scene = cv2.equalizeHist(small_scene)
        small_template = cv2.equalizeHist(small_template)

        angles = np.arange(0, 360, coarse_step)
        all_boxes, all_scores, all_angles = [], [], []

        for angle in angles:
            M, (new_w, new_h) = rotate_image_keep_all(small_template, angle)
            rotated_temple = cv2.warpAffine(
                small_template, M, (new_w, new_h),
                flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
            )
            if rotated_temple.shape[0] > small_scene.shape[0] or rotated_temple.shape[1] > small_scene.shape[1]:
                continue

            res = cv2.matchTemplate(small_scene, rotated_temple, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                all_boxes.append([pt[0], pt[1], pt[0] + new_w, pt[1] + new_h])
                all_scores.append(float(res[pt[1], pt[0]]))
                all_angles.append(angle)

        if not all_boxes:
            print("❌ Không có vùng vượt ngưỡng coarse.")
            return []

        keep = cv2.dnn.NMSBoxes(all_boxes, all_scores, score_threshold=threshold, nms_threshold=0.3)
        if len(keep) == 0:
            return []

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
        refine_results = []

        for candidate in coarse_candidates:
            x1, y1, x2, y2 = candidate["box"]
            angle_c = candidate["angle"]

            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(scene.shape[1], x2 + pad), min(scene.shape[0], y2 + pad)
            roi = scene[y1p:y2p, x1p:x2p]
            if roi.size == 0:
                continue

            best_local = None
            local_angles = np.arange(angle_c - 15, angle_c + 15 + 1, refine_step)
            for a in local_angles:
                M, (new_w, new_h) = rotate_image_keep_all(template, a)
                rotated_t = cv2.warpAffine(template, M, (new_w, new_h),
                                           flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
                if rotated_t.shape[0] > roi.shape[0] or rotated_t.shape[1] > roi.shape[1]:
                    continue

                res = cv2.matchTemplate(roi, rotated_t, cv2.TM_CCOEFF_NORMED) 
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val < threshold:
                    continue

                abs_loc = (max_loc[0] + x1p, max_loc[1] + y1p)
                if (best_local is None) or (max_val > best_local["score"]):
                    best_local = {
                        "shape": "oriented_box",
                        "box": [abs_loc[0], abs_loc[1],
                                abs_loc[0] + new_w, abs_loc[1] + new_h],
                        "angle": a,
                        "score": max_val
                    }

            if best_local:
                refine_results.append(best_local)

        if not refine_results:
            print("❌ Không có refine result.")
            return []

        boxes = [r["box"] for r in refine_results]
        scores = [r["score"] for r in refine_results]
        keep = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=threshold, nms_threshold=0.3)
        if len(keep) == 0:
            return []

        keep = sorted(keep.flatten(), key=lambda i: scores[i], reverse=True)[:max_objects]

        print(f"✅ [REFINE] giữ lại {len(keep)} đối tượng.")
        print(f"⏱ Tổng thời gian: {time.time() - t0:.2f}s")

        return [refine_results[i] for i in keep]


