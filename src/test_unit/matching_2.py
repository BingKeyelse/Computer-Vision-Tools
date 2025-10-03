import cv2
import numpy as np
import matplotlib.pyplot as plt

def match_and_draw_box(img1_path, img2_path, min_match_count=10):
    # Đọc ảnh
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)  # object
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)  # scene

    if img1 is None or img2 is None:
        raise ValueError("Không thể đọc ảnh. Kiểm tra lại đường dẫn.")

    # Dùng ORB (có thể thay bằng SIFT nếu cài opencv-contrib)
    orb = cv2.ORB_create(1000)

    # Detect keypoints và descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # BFMatcher với crossCheck = False để dùng knnMatch
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Lấy 2 nearest neighbors cho mỗi descriptor
    matches = bf.knnMatch(des1, des2, k=2)

    # Lọc theo Lowe’s ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(f"Tìm được {len(good)} good matches")

    if len(good) > min_match_count:
        # Lấy tọa độ điểm khớp
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Tính Homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Lấy shape ảnh gốc
        h, w = img1.shape

        # 4 điểm góc ảnh 1
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)

        # Biến đổi sang ảnh 2
        dst = cv2.perspectiveTransform(pts, M)

        # Vẽ box lên ảnh scene (img2 màu)
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        cv2.polylines(img2_color, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)

        # Vẽ matches để debug
        matched_img = cv2.drawMatches(img1, kp1, img2_color, kp2, good, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Hiển thị
        plt.figure(figsize=(14, 7))
        plt.imshow(matched_img)
        plt.axis('off')
        plt.show()
    else:
        print("Không đủ matches để tìm Homography")

match_and_draw_box(r"src\data\sample\1.jpg", r"src\data\sample\sample.jpg", min_match_count=15)

