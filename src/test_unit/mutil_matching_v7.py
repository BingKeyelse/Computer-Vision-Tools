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
                        threshold=0.7, max_candidates=10, max_objects=5, pad=20) ->  list | None:
    """
    ## Matching 2 pharse with 1st: coarse and 2nd: refine
    - Args: 
        - template: ảnh temple ở dạng gray
        - scene: ảnh gốc ở dạng gray
        - coarse_scale: tỉ lệ scale thô mà bạn muốn
        - coarse_step: step xoay với scale thô với góc 360
        - refine_step: step góc quét tinh sau khi biết được góc tương đối cần quét rồi
        - threshold: ngưỡng matching accept
        - max_candidates: refine số vùng tốt nhất sau coarse
        - max_objects: số đối tượng tối đa sau khi refine + NMS
        - pad:mở rộng ROI quanh box coarse trước khi refine để tránh cắt nhầm
    """
    
    t0 = time.time()

    # --- COARSE PHASE ---
    print("🌀 [COARSE] scanning...")
    small_scene = cv2.resize(scene, (0,0), fx=coarse_scale, fy=coarse_scale)
    small_template = cv2.resize(template, (0,0), fx=coarse_scale, fy=coarse_scale)

    # Tạo môt list numpy với 360 độ/ step chưa các góc mong muốn
    angles = np.arange(0, 360, coarse_step)

    # List để chứa box, điểm tin cậy, góc kèm theo
    all_boxes, all_scores, all_angles = [], [], []

    # Quét lần lượt các góc
    for angle in angles:
        M, (new_w, new_h) = rotate_image_keep_all(small_template, angle)
        rotated_temple = cv2.warpAffine(small_template, M, (new_w, new_h),
                                 flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
        
        if rotated_temple.shape[0] <= small_scene.shape[0] and rotated_temple.shape[1] <= small_scene.shape[1]:

            # Matching
            res = cv2.matchTemplate(small_scene, rotated_temple, cv2.TM_CCOEFF_NORMED)

            # So sánh với ngưỡng để lấy ra các giá trị box, score, angle tương ứng
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                all_boxes.append([pt[0], pt[1], pt[0]+new_w, pt[1]+new_h])
                all_scores.append(float(res[pt[1], pt[0]]))
                all_angles.append(angle)

    if not all_boxes:
        print("❌ Không có vùng vượt ngưỡng trong coarse scan.")
        return None

    # NMS lọc vùng trùng. Nếu chồng lấn quá 30% thì loại bỏ luôn
        ##  Chỉ giữ lại score ≥ score_threshold
    keep = cv2.dnn.NMSBoxes(all_boxes, all_scores, score_threshold=threshold, nms_threshold=0.3)
    if len(keep) == 0:
        print("❌ Không còn box sau NMS (coarse).")
        return None

    # Vì NMSBoxes tra ra kiểu dữ liệu shape (N, 1) nên phải flatten
    keep = keep.flatten()
    # Lấy k đối tượng max_candidates tốt nhất
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

    print(f"✅ [Scan COARSE] giữ lại {len(coarse_candidates)} vùng nghi ngờ để refine")

    # --- REFINE PHASE ---
    print("🎯 [REFINE] scanning around each candidate...")
    refine_results = []

    # Scan refine area
    for candidate in coarse_candidates:
        x1, y1, x2, y2 = candidate["box"]
        angle_c = candidate["angle"]

        # Thêm padding để tránh cắt biên
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(scene.shape[1], x2 + pad)
        y2p = min(scene.shape[0], y2 + pad)
        roi = scene[y1p:y2p, x1p:x2p]
        if roi.size == 0:
            continue

        best_local = None
        # Xử lý 15 độ mỗi bên từ angle nhận diện được ở trên
        local_angles = np.arange(angle_c - 15, angle_c + 15 + 1, refine_step)
        for a in local_angles:
            M, (new_w, new_h) = rotate_image_keep_all(template, a)
            rotated_temple = cv2.warpAffine(template, M, (new_w, new_h),
                                     flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
            
            # Check temple có size nhở hơn target thì mới chạy
            if rotated_temple.shape[0] <= roi.shape[0] and rotated_temple.shape[1] <= roi.shape[1]:
                # Matching 
                    ## Roi: Vùng trong ảnh chính
                    ## rotated_temple: ảnh xoay 30 độ quanh góc detect được
                res = cv2.matchTemplate(roi, rotated_temple, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                # Lấy giá trị max_val (score) lớn nhất nhé
                if max_val >= threshold:
                    abs_loc = (max_loc[0] + x1p, max_loc[1] + y1p)
                    # Nếu lớn hơn best local score hiện tại thì mới được cập nhập trong 30 độ check này
                    if (best_local is None) or (max_val > best_local["score"]):
                        best_local = {
                            "box": [abs_loc[0], abs_loc[1],
                                    abs_loc[0]+new_w, abs_loc[1]+new_h],
                            "angle": a,
                            "score": max_val
                        }
        
        # Lấy giá trị tốt nhất của từng coarse_candidates
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

    # --- Hiển thị oriented boxes ---
    scene_color = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    h_t, w_t = template.shape[:2]  # kích thước thật của template

    for index in keep: # Lấy index của keep ra 
        r = refine_results[index]
        x1, y1, x2, y2 = r["box"]
        angle = r["angle"]
        score = r["score"]

        # ---- Tính lại vị trí thực tế của template gốc trong ảnh xoay ----
        # Xoay template gốc để biết offset
        M_rot, (new_w, new_h) = rotate_image_keep_all(template, angle)

        # Tính vị trí của template gốc (w_t,h_t) khi chưa xoay
        corners_t = np.array([
            [0, 0],
            [w_t, 0],
            [w_t, h_t],
            [0, h_t]
        ], dtype=np.float32)

        ones = np.ones((4, 1), dtype=np.float32)
        corners_h = np.hstack([corners_t, ones])
        rotated_t = (M_rot @ corners_h.T).T  # Toạ độ template gốc trong canvas và đã được xoay với 4 điểm góc hình chữ nhật đã xoay rồi

        # Lấy minX, minY để dịch về vị trí match
        offset_x = rotated_t[:, 0].min()
        offset_y = rotated_t[:, 1].min()

        # Dịch các góc về vị trí thật trong scene
        rotated_in_scene = rotated_t - [offset_x, offset_y] + [x1, y1]
        rotated_in_scene = rotated_in_scene.astype(np.int32)

        # --- Vẽ polygon ---
        cv2.polylines(scene_color, [rotated_in_scene], isClosed=True, color=(0, 255, 0), thickness=2)

        # Tính tâm trung bình
        cx, cy = np.mean(rotated_in_scene, axis=0).astype(int)
        cv2.putText(scene_color, f"angle: {angle:.1f}deg and score: {score:.2f}",
                    (int(cx), int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    print(f"⏱ Tổng thời gian: {time.time() - t0:.2f}s")
    cv2.imshow("Coarse-Refine Detection", scene_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return [refine_results[i] for i in keep]

# --- Test ---
scene = cv2.imread(r"src\data\sample\5.jpg", cv2.IMREAD_GRAYSCALE)
template = cv2.imread(r"src\data\sample\sample.jpg", cv2.IMREAD_GRAYSCALE)

# scene = cv2.imread(r"src\data\sample\mutil\1.jpg", cv2.IMREAD_GRAYSCALE)
# template = cv2.imread(r"src\data\sample\mutil\temple.jpg", cv2.IMREAD_GRAYSCALE)

template = cv2.resize(template, (0,0), fx=0.5, fy=0.5)
scene = cv2.resize(scene, (0,0), fx=0.5, fy=0.5)

results = coarse_refine_match(template, scene,
                              coarse_scale=0.2,
                              coarse_step=5,
                              refine_step=1,
                              threshold=0.65,
                              max_candidates=15,
                              max_objects=5)

if results:
    print("\n📋 Kết quả cuối:")
    for i, r in enumerate(results, 1):
        print(f"{i:02d}. angle={r['angle']}°, score={r['score']:.3f}, box={r['box']}")
