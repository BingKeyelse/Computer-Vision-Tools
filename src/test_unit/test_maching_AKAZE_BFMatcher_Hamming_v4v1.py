#Với AKAZE thì sẽ dùng BFMatcher + Hamming, còn với SURF/SIFT thì vẫn dùng FLANN.

import cv2
import numpy as np
import matplotlib.pyplot as plt

def match_and_draw_box(template_path, scene_path, use_akaze=True):
    # Đọc ảnh
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    scene = cv2.imread(scene_path, cv2.IMREAD_COLOR)

    # Resize nhỏ 1/2 cho dễ hiển thị
    template = cv2.resize(template, (template.shape[1]//2, template.shape[0]//2))
    scene = cv2.resize(scene, (scene.shape[1]//2, scene.shape[0]//2))

    # Chọn detector
    if use_akaze:
        detector = cv2.AKAZE_create()
    else:
        detector = cv2.SIFT_create()   # hoặc cv2.xfeatures2d.SURF_create()

    # Tìm keypoint và descriptor
    kp1, des1 = detector.detectAndCompute(template, None)
    kp2, des2 = detector.detectAndCompute(scene, None)

    if des1 is None or des2 is None:
        print("Không tìm thấy đặc trưng nào.")
        return

    # Matching
    if use_akaze:
        # AKAZE -> descriptor nhị phân => dùng Hamming
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
    else:
        # SIFT/SURF -> descriptor float => dùng FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

    # Ratio test (lọc match tốt)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) >= 4:
        # Lấy 20 match tốt nhất để ổn định hơn
        good = sorted(good, key=lambda x: x.distance)[:20]

        # Lấy tọa độ keypoint tương ứng
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Tính ma trận Homography (phép biến đổi perspective)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            print("Homography matrix:\n", M)

            # Tính góc xoay (ước lượng từ ma trận)
            angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
            print(f"Ước lượng góc xoay: {angle:.2f}°")

            # Vẽ bounding box theo homography
            h, w = template.shape[:2]
            pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            dst_box = cv2.perspectiveTransform(pts, M)

            # Nếu ảnh là grayscale → chuyển sang BGR để vẽ màu
            if len(scene.shape) == 2:
                scene_color = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
            else:
                scene_color = scene.copy()

            # Vẽ khung đối tượng tìm được
            cv2.polylines(scene_color, [np.int32(dst_box)], True, (0, 255, 0), 3)

            # --- Thêm phần warp template sang scene ---
            aligned = cv2.warpPerspective(template, M, (scene.shape[1], scene.shape[0]))
            cv2.imshow("Warped Template", aligned)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Hiển thị match để debug
            matched_img = cv2.drawMatches(
                template, kp1, scene_color, kp2, good, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            plt.figure(figsize=(14, 7))
            plt.imshow(matched_img[..., ::-1])  # BGR → RGB
            plt.axis("off")
            plt.title(f"Matching thành công với {len(good)} điểm tốt.")
            plt.show()
        else:
            print("Không tính được homography.")
    else:
        print("Không đủ match tốt để tính homography.")

# Test
match_and_draw_box(r"src\data\sample\5.jpg", r"src\data\sample\sample_1.jpg", use_akaze=False)
