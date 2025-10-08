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


class BoxMatcher(BaseMatcher):  
    def __init__(self, image_path, data):
        super().__init__(image_path, data)
        self.template = None

    def load_template(self):
        _, (x1, y1), (x2, y2) = self.data
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        self.template = img[int(y1):int(y2), int(x1):int(x2)]

    def match(self, scene,
              coarse_scale=0.3,
              coarse_step=10,
              refine_step=2,
              threshold=0.7,
              max_candidates=10,
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

