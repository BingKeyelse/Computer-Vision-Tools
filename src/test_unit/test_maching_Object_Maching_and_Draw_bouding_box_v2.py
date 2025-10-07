import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def match_and_draw_box(img1_path, img2_path, min_match_count=10):
    time_start= time.time()
    # Đọc ảnh
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)  # object
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)  # scene

    if img1 is None or img2 is None:
        raise ValueError("Không thể đọc ảnh. Kiểm tra lại đường dẫn.")

    # Dùng ORB (có thể thay bằng SIFT nếu cài opencv-contrib)
    orb = cv2.ORB_create(100)

    # Detect keypoints và descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # BFMatcher với crossCheck = False để dùng knnMatch
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Lấy 2 nearest neighbors cho mỗi descriptor
    matches = bf.knnMatch(des1, des2, k=2)
    print(matches[0])
    for i, (m, n) in enumerate(matches[1:2]):  # chỉ in 5 cặp đầu tiên
        print(f"Match {i}:")
        print(f"  Best match: queryIdx={m.queryIdx}, trainIdx={m.trainIdx}, distance={m.distance}")
        print(f"  Second best: queryIdx={n.queryIdx}, trainIdx={n.trainIdx}, distance={n.distance}")

    # Lọc theo Lowe’s ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance: # 0,75-0.8 thus paper
            good.append(m)
    print(len(good))

    print(f"Tìm được {len(good)} good matches")
    print(f'Tổng thời gian xử lý là {time.time()- time_start}')

    top_n = 6  # chỉ dùng 8 match tốt nhất

    if len(good) > min_match_count:
        # Lấy top_n match tốt nhất
        good_top = sorted(good, key=lambda x: x.distance)[:top_n]

        # Lấy tọa độ điểm khớp
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_top]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_top]).reshape(-1, 1, 2)

        # Tính Homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # M bản chất là ma trận 3x3 
        
        if M is None:
            print("Homography không tìm được")
            return

        # Tách rotation + translation
        H_affine = M[:2, :2]
        tx, ty = M[0,2], M[1,2]

        U, _, Vt = np.linalg.svd(H_affine)
        R = U @ Vt   # pure rotation

        # Chuẩn hóa bỏ scale (chỉ lấy rotation)
        # R = H_affine / np.linalg.norm(H_affine[:,0])

        # 4 góc ảnh sample
        h, w = img1.shape
        pts = np.float32([[0,0],[w,0],[w,h],[0,h]])

        # Áp rotation + translation
        dst_box = np.dot(pts, R.T) + np.array([tx, ty])

        # Vẽ bounding box
        dst_box = dst_box.reshape(-1,1,2).astype(np.int32)
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        cv2.polylines(img2_color, [dst_box], True, (0,255,0), 3, cv2.LINE_AA)

        # Vẽ matches top_n để debug
        matched_img = cv2.drawMatches(img1, kp1, img2_color, kp2, good_top, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Hiển thị
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 7))
        plt.imshow(matched_img)
        plt.axis('off')
        plt.show()
    else:
        print("Không đủ matches để tìm Homography")

match_and_draw_box(r"src\data\sample\sample_1.jpg", r"src\data\sample\2.jpg", min_match_count=15)
