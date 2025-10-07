import cv2
import numpy as np
import time

def rotate_image_keep_all(img, angle, borderValue=(255,255,255)):
    """Rotate image around center but keep full content (expanded canvas)."""
    (h, w) = img.shape[:2]
    angle_rad = np.deg2rad(angle)
    cos, sin = abs(np.cos(angle_rad)), abs(np.sin(angle_rad))
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    M[0,2] += (new_w - w) / 2
    M[1,2] += (new_h - h) / 2
    return M, (new_w, new_h)

def coarse_refine_match(template, scene,
                        coarse_scale=0.5, coarse_step=10, refine_step=2,
                        threshold=0.7, max_candidates=10, max_objects=5, pad=20):
    t0 = time.time()

    # --- COARSE PHASE ---
    print("🌀 [COARSE] scanning...")
    small_scene = cv2.resize(scene, (0,0), fx=coarse_scale, fy=coarse_scale)
    small_template = cv2.resize(template, (0,0), fx=coarse_scale, fy=coarse_scale)

    angles = np.arange(0, 360, coarse_step)
    all_boxes, all_scores, all_angles = [], [], []

    for angle in angles:
        M, (new_w, new_h) = rotate_image_keep_all(small_template, angle)
        rot_tpl = cv2.warpAffine(small_template, M, (new_w, new_h),
                                 flags=cv2.INTER_LINEAR, borderValue=(255,255,255))

        res = cv2.matchTemplate(small_scene, rot_tpl, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            all_boxes.append([pt[0], pt[1], pt[0]+new_w, pt[1]+new_h])
            all_scores.append(float(res[pt[1], pt[0]]))
            all_angles.append(angle)

    if not all_boxes:
        print("❌ Không có vùng vượt ngưỡng trong coarse scan.")
        return None

    # NMS lọc vùng trùng
    keep = cv2.dnn.NMSBoxes(all_boxes, all_scores, score_threshold=threshold, nms_threshold=0.3)
    if len(keep) == 0:
        print("❌ Không còn box sau NMS (coarse).")
        return None

    keep = keep.flatten()
    keep = sorted(keep, key=lambda i: all_scores[i], reverse=True)[:max_candidates]

    # Chuyển về ảnh gốc
    coarse_candidates = []
    for i in keep:
        x1, y1, x2, y2 = np.array(all_boxes[i]) / coarse_scale
        coarse_candidates.append({
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "angle": all_angles[i],
            "score": all_scores[i]
        })

    print(f"✅ [COARSE] giữ lại {len(coarse_candidates)} vùng nghi ngờ để refine")

    # --- REFINE PHASE ---
    print("🎯 [REFINE] scanning around each candidate...")
    refine_results = []
    for c in coarse_candidates:
        x1, y1, x2, y2 = c["box"]
        angle_c = c["angle"]

        # Thêm padding để tránh cắt biên
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(scene.shape[1], x2 + pad)
        y2p = min(scene.shape[0], y2 + pad)
        roi = scene[y1p:y2p, x1p:x2p]
        if roi.size == 0:
            continue

        best_local = None
        local_angles = np.arange(angle_c - 5, angle_c + 5 + 1, refine_step)
        for a in local_angles:
            M, (new_w, new_h) = rotate_image_keep_all(template, a)
            rot_tpl = cv2.warpAffine(template, M, (new_w, new_h),
                                     flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
            res = cv2.matchTemplate(roi, rot_tpl, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val >= threshold:
                abs_loc = (max_loc[0] + x1p, max_loc[1] + y1p)
                if (best_local is None) or (max_val > best_local["score"]):
                    best_local = {
                        "box": [abs_loc[0], abs_loc[1],
                                abs_loc[0]+new_w, abs_loc[1]+new_h],
                        "angle": a,
                        "score": max_val
                    }
        if best_local:
            refine_results.append(best_local)

    if not refine_results:
        print("❌ Không phát hiện được gì ở refine.")
        return None

    # --- NMS sau refine ---
    boxes = [r["box"] for r in refine_results]
    scores = [r["score"] for r in refine_results]
    keep = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=threshold, nms_threshold=0.3)
    if len(keep) == 0:
        print("❌ Không còn box sau NMS (refine).")
        return None

    keep = keep.flatten()
    keep = sorted(keep, key=lambda i: scores[i], reverse=True)[:max_objects]

    print(f"✅ [REFINE] giữ lại {len(keep)} đối tượng cuối cùng.")

    # --- Hiển thị ---
    scene_color = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(scene_color, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(scene_color, f"{refine_results[i]['angle']}° {scores[i]:.2f}",
                    (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    print(f"⏱ Tổng thời gian: {time.time() - t0:.2f}s")
    cv2.imshow("Coarse-Refine Detection", scene_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return [refine_results[i] for i in keep]

# --- Test ---
scene = cv2.imread(r"src\data\sample\mutil\6.jpg", cv2.IMREAD_GRAYSCALE)
template = cv2.imread(r"src\data\sample\mutil\temple.jpg", cv2.IMREAD_GRAYSCALE)

template = cv2.resize(template, (0,0), fx=0.5, fy=0.5)
scene = cv2.resize(scene, (0,0), fx=0.5, fy=0.5)

results = coarse_refine_match(template, scene,
                              coarse_scale=0.35,
                              coarse_step=5,
                              refine_step=1,
                              threshold=0.8,
                              max_candidates=8,
                              max_objects=5)

if results:
    print("\n📋 Kết quả cuối:")
    for i, r in enumerate(results, 1):
        print(f"{i:02d}. angle={r['angle']}°, score={r['score']:.3f}, box={r['box']}")
