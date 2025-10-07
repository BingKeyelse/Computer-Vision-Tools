#Với AKAZE thì sẽ dùng BFMatcher + Hamming, còn với SURF/SIFT thì vẫn dùng FLANN.

import cv2
import numpy as np

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

    # Ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Homography để vẽ box
    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = template.shape[:2]
        pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)

        scene = cv2.polylines(scene, [np.int32(dst)], True, (0,255,0), 3)
        print(f"Matching thành công với {len(good)} điểm tốt.")
    else:
        print("Không đủ match tốt để tính homography.")

    # Hiển thị
    result = cv2.drawMatches(template, kp1, scene, kp2, good, None, flags=2)
    cv2.imshow("Matching", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test
match_and_draw_box(r"src\data\sample\1.jpg", r"src\data\sample\sample_1.jpg", use_akaze=False)
