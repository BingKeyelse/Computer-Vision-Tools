import cv2
import numpy as np
import time

# --- Xoay ảnh quanh tâm ---
def rotate_image_keep_all(img: np.ndarray, angle: float, borderValue=(0, 0, 0)):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=borderValue)
    return rotated, (h, w)

# --- Matching theo hướng coarse → refine ---
def circle_template_match(scene_path, template_path,
                          coarse_step=20, refine_step=2,
                          coarse_scale=0.5, threshold=0.7,
                          max_candidates=10, max_objects=3):
    

    # 1️⃣ Load ảnh
    scene = cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if scene is None or template is None:
        raise ValueError("Không đọc được ảnh scene hoặc template.")

    print(f"Scene: {scene.shape}, Template: {template.shape}")

    # 2️⃣ Giảm kích thước để chạy coarse nhanh hơn
    small_scene = cv2.resize(scene, (0, 0), fx=coarse_scale, fy=coarse_scale)
    small_template = cv2.resize(template, (0, 0), fx=coarse_scale, fy=coarse_scale)
 

    print("🌀 [COARSE] scanning...")
    t0 = time.time()

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

    # --- Tạo mask tròn từ small_template (1-channel) ---
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

    results = [refine_results[i] for i in keep]
    print(f"✅ [REFINE] giữ lại {len(results)} đối tượng. ⏱ {time.time()-t0:.2f}s")

    # 7️⃣ Hiển thị kết quả
    vis = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    for r in results:
        x1, y1, x2, y2 = map(int, r["box"])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        # Vẽ hình tròn
        cv2.circle(vis, (cx, cy), radius, (0, 255, 0), 3)
        cv2.putText(vis, f"{r['angle']:.1f}° ({r['score']:.2f})",
                    (cx - radius, cy - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Result", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # return results

# ----------------------------
# 🧩 Test chương trình
# ----------------------------
if __name__ == "__main__":
    scene_path = r"images\sample_cam1.png"
    template_path = r"images\saved_sample.png"
    circle_template_match(scene_path, template_path)
