import cv2
import numpy as np

def match_and_draw_box(template_path, scene_path):
    # Đọc ảnh
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    scene = cv2.imread(scene_path, cv2.IMREAD_COLOR)

    if template is None or scene is None:
        print("[ERROR] Không load được ảnh.")
        return
    
    # Resize còn 1/2
    template = cv2.resize(template, (template.shape[1]//2, template.shape[0]//2))
    scene = cv2.resize(scene, (scene.shape[1]//2, scene.shape[0]//2))


    # Tạo SIFT detector
    sift = cv2.SIFT_create()

    # Trích đặc trưng
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(scene, None)

    # Match bằng BFMatcher + knn
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(f"[INFO] Số match tốt: {len(good)}")

    # Tính Homography nếu đủ điểm
    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            h, w = template.shape[:2]
            pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, M)

            # Vẽ box lên ảnh gốc
            scene = cv2.polylines(scene, [np.int32(dst)], True, (0,255,0), 3)
        else:
            print("[WARN] Không tính được Homography.")
    else:
        print("[WARN] Không đủ match để tính Homography.")

    # Hiển thị kết quả
    cv2.imshow("Matching Result", scene)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Gọi thử
match_and_draw_box( r"src\data\sample\sample_1.jpg", r"src\data\sample\1.jpg")
