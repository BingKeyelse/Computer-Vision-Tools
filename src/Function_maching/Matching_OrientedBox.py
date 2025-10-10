from libs import*

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

def extract_oriented_object(img, start, end, angle_rad):
    """
    Chuẩn hóa vùng oriented box về phương nằm ngang.
    - start, end: 2 điểm đối diện nhau (đường chéo)
    - angle_rad: góc nghiêng (radian)
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


class OrientedBoxMatcher(BaseMatcher):  
    def __init__(self, temple_path, data, scale):
        super().__init__(temple_path, data, scale)
        self.template = None

    def load_template(self):
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

    def match(self, scene,
              coarse_scale=0.3,
              coarse_step=10,
              refine_step=2,
              threshold=0.7,
              max_candidates=15,
              max_objects=5,
              pad=20):
        """
        Dò tìm template hình hộp chữ nhật trong scene với coarse→refine
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

        #### Test
        # # --- Hiển thị oriented boxes ---
        # scene_color = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
        # h_t, w_t = template.shape[:2]  # kích thước thật của template

        # for index in keep: # Lấy index của keep ra 
        #     r = refine_results[index]
        #     x1, y1, x2, y2 = r["box"]
        #     angle = r["angle"]
        #     score = r["score"]

        #     # ---- Tính lại vị trí thực tế của template gốc trong ảnh xoay ----
        #     # Xoay template gốc để biết offset
        #     M_rot, (new_w, new_h) = rotate_image_keep_all(template, angle)

        #     # Tính vị trí của template gốc (w_t,h_t) khi chưa xoay
        #     corners_t = np.array([
        #         [0, 0],
        #         [w_t, 0],
        #         [w_t, h_t],
        #         [0, h_t]
        #     ], dtype=np.float32)

        #     ones = np.ones((4, 1), dtype=np.float32)
        #     corners_h = np.hstack([corners_t, ones])
        #     rotated_t = (M_rot @ corners_h.T).T  # Toạ độ template gốc trong canvas và đã được xoay với 4 điểm góc hình chữ nhật đã xoay rồi

        #     # Lấy minX, minY để dịch về vị trí match
        #     offset_x = rotated_t[:, 0].min()
        #     offset_y = rotated_t[:, 1].min()

        #     # Dịch các góc về vị trí thật trong scene
        #     rotated_in_scene = rotated_t - [offset_x, offset_y] + [x1, y1]
        #     rotated_in_scene = rotated_in_scene.astype(np.int32)

        #     # --- Vẽ polygon ---
        #     cv2.polylines(scene_color, [rotated_in_scene], isClosed=True, color=(0, 255, 0), thickness=2)

        #     # Tính tâm trung bình
        #     cx, cy = np.mean(rotated_in_scene, axis=0).astype(int)
        #     cv2.putText(scene_color, f"angle: {angle:.1f}deg and score: {score:.2f}",
        #                 (int(cx), int(cy) - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # print(f"⏱ Tổng thời gian: {time.time() - t0:.2f}s")
        # cv2.imshow("Coarse-Refine Detection", scene_color)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        print(f"✅ [REFINE] giữ lại {len(keep)} đối tượng.")
        print(f"⏱ Tổng thời gian: {time.time() - t0:.2f}s")

        return [refine_results[i] for i in keep]

## Cách dùng
# scene = cv2.imread("scene.jpg", cv2.IMREAD_GRAYSCALE)
# matcher = BoxMatcher("template_source.jpg", ('box', (100, 200), (400, 600)))
# matcher.load_template()
# results = matcher.match(scene)
# for r in results:
#     print(r)

