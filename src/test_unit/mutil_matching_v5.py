import cv2
import numpy as np
import time

def rotate_image_keep_all(img, angle, borderValue=(255,255,255)):
    (h, w) = img.shape[:2]
    angle_rad = np.deg2rad(angle)
    cos = abs(np.cos(angle_rad))
    sin = abs(np.sin(angle_rad))
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    M[0,2] += (new_w - w) / 2
    M[1,2] += (new_h - h) / 2
    return M, (new_w, new_h)

def match_template_multi_rotated_fast(template, scene, step=10, threshold=0.7, max_objects=5, top_k_per_angle=5):
    t0 = time.time()
    all_boxes, all_scores, all_angles = [], [], []

    # Resize nhỏ để tăng tốc
    template = cv2.resize(template, (0,0), fx=0.5, fy=0.5)
    scene = cv2.resize(scene, (0,0), fx=0.5, fy=0.5)

    angles = np.arange(0, 360, step)
    print(f"🔄 Quét {len(angles)} góc...")

    for angle in angles:
        M, (new_w, new_h) = rotate_image_keep_all(template, angle)
        rot_tpl = cv2.warpAffine(template, M, (new_w, new_h),
                                 flags=cv2.INTER_LINEAR, borderValue=(255,255,255))

        # Template matching
        res = cv2.matchTemplate(scene, rot_tpl, cv2.TM_CCOEFF_NORMED)
        res_copy = res.copy()

        # --- lấy top K điểm tốt nhất cho mỗi góc ---
        for _ in range(top_k_per_angle):
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_copy)
            if max_val < threshold:
                break  # dừng nếu không còn điểm nào đủ tốt

            x1, y1 = max_loc
            x2, y2 = x1 + new_w, y1 + new_h
            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(max_val)
            all_angles.append(angle)

            # Xóa vùng vừa chọn (để không bị chọn trùng)
            cv2.rectangle(res_copy, (x1, y1), (x2, y2), 0, -1)

        print(f"angle={angle:3d}° -> {len(all_boxes)} tổng match sau góc này")

    if not all_boxes:
        print("❌ Không phát hiện được đối tượng nào vượt ngưỡng.")
        return None

    # --- Áp dụng NMS ---
    keep = cv2.dnn.NMSBoxes(all_boxes, all_scores,
                            score_threshold=threshold, nms_threshold=0.3)

    if len(keep) == 0:
        print("❌ Không còn box sau NMS.")
        return None

    keep = keep.flatten()
    print(f"\n✅ Giữ lại {len(keep)} box sau NMS")

    # --- Sắp xếp theo score giảm dần ---
    keep = sorted(keep, key=lambda i: all_scores[i], reverse=True)[:max_objects]
    print(f"📦 Chỉ lấy top {len(keep)} đối tượng có score cao nhất")

    # --- Vẽ kết quả ---
    scene_color = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    for i in keep:
        x1, y1, x2, y2 = all_boxes[i]
        angle = all_angles[i]
        score = all_scores[i]
        cv2.rectangle(scene_color, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(scene_color, f"{angle}° {score:.2f}",
                    (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    print(f"⏱ Thời gian: {time.time() - t0:.2f} giây")
    cv2.imshow("Detected Objects", scene_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Trả về danh sách kết quả
    results = [ {
        'angle': all_angles[i],
        'score': all_scores[i],
        'box': all_boxes[i]
    } for i in keep ]

    return results


# --- Test ---
scene = cv2.imread(r"src\data\sample\2.jpg", cv2.IMREAD_GRAYSCALE)
template = cv2.imread(r"src\data\sample\sample.jpg", cv2.IMREAD_GRAYSCALE)

detections = match_template_multi_rotated_fast(template, scene, step=5, threshold=0.8, max_objects=1, top_k_per_angle=3)

if detections:
    print("\n📋 Danh sách kết quả:")
    for i, det in enumerate(detections, 1):
        print(f"{i:02d}. angle={det['angle']}°, score={det['score']:.3f}, box={det['box']}")
