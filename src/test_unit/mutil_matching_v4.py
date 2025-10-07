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

def match_template_multi_rotated(template, scene, step=10, threshold=0.7, max_objects=5):
    t0 = time.time()
    all_boxes, all_scores, all_angles = [], [], []

    # Resize cho nhanh
    template = cv2.resize(template, (0,0), fx=0.4, fy=0.4)
    scene = cv2.resize(scene, (0,0), fx=0.4, fy=0.4)

    angles = np.arange(0, 360, step)
    print(f"🔄 Quét {len(angles)} góc...")

    for angle in angles:
        M, (new_w, new_h) = rotate_image_keep_all(template, angle)
        rot_tpl = cv2.warpAffine(template, M, (new_w, new_h),
                                 flags=cv2.INTER_LINEAR, borderValue=(255,255,255))

        res = cv2.matchTemplate(scene, rot_tpl, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        k=0

        for pt in zip(*loc[::-1]):
            all_boxes.append([pt[0], pt[1], pt[0] + new_w, pt[1] + new_h])
            all_scores.append(float(res[pt[1], pt[0]]))
            k=float(res[pt[1], pt[0]])
            all_angles.append(angle)

        print(f"angle={angle:3d}° -> {len(loc[0])} matches with {k}")
    
    print(str(len(all_angles))+"sadfsafds")

    if not all_boxes:
        print("❌ Không phát hiện được đối tượng nào vượt ngưỡng.")
        return None

    # --- Áp dụng NMS ---
    keep = cv2.dnn.NMSBoxes(all_boxes, all_scores,
                            score_threshold=threshold,
                            nms_threshold=0.3)

    if len(keep) == 0:
        print("❌ Không còn box sau NMS.")
        return None

    keep = keep.flatten()
    print(f"\n✅ Giữ lại {len(keep)} box sau NMS")

    # --- Sắp xếp theo score giảm dần ---
    keep = sorted(keep, key=lambda i: all_scores[i], reverse=True)

    # --- Giới hạn số lượng kết quả ---
    keep = keep[:max_objects]
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
    results = [{
        'angle': all_angles[i],
        'score': all_scores[i],
        'box': all_boxes[i]
    } for i in keep]

    return results


# --- Test ---
scene = cv2.imread(r"src\data\sample\1.jpg", cv2.IMREAD_GRAYSCALE)
template = cv2.imread(r"src\data\sample\sample.jpg", cv2.IMREAD_GRAYSCALE)

detections = match_template_multi_rotated(template, scene, step=5, threshold=0.8, max_objects=2)

if detections:
    print("\n📋 Danh sách kết quả:")
    for i, det in enumerate(detections, 1):
        print(f"{i:02d}. angle={det['angle']}°, score={det['score']:.3f}, box={det['box']}")
