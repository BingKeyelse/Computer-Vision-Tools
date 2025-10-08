import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def match_and_draw_box_rotated(img1_path, img2_path, max_matches=50):
    """
    1. Đọc 2 ảnh grayscale
    2. Dùng ORB tìm keypoints và descriptors
    3. Match points bằng BFMatcher
    4. Vẽ top match points
    5. Tính homography
    6. Vẽ rotated bounding box chính xác quanh object trên target
    """
    # Time start
    time_start= time.time()
    # Đọc ảnh grayscale
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError("Không thể đọc ảnh. Kiểm tra lại đường dẫn.")

    # Tạo ORB detector
    orb = cv2.ORB_create()

    # Tìm keypoints và descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Brute-Force matcher với Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des1, des2)
    print(f"Số điểm match: {len(matches)}")

    # Sắp xếp matches theo khoảng cách
    matches = sorted(matches, key=lambda x: x.distance)
    print(f'Tổng thời gian xử lý là {time.time()- time_start}')

    # Vẽ các top match points
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:max_matches], None, flags=2)
    plt.figure(figsize=(12,6))
    plt.imshow(matched_img)
    plt.axis('off')
    plt.title("Top Match Points")
    plt.show()

    # --- Tính homography và vẽ rotated bounding box ---
    if len(matches) >= 4:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        # Tính homography với RANSAC để tính ma trận 3x3 H để map cho tất cả keypoints.sample -> target
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # H chứa tất cả các biến đổi planar
        # Translation (dịch chuyển)

        # Rotation (xoay trên plane của object)

        # Scale (phóng to/thu nhỏ)

        # Skew / perspective (biến dạng do camera nghiêng)

        matches_mask = mask.ravel().tolist()
        inlier_dst_pts = dst_pts[np.array(matches_mask) == 1]

        # Nếu đủ inliers
        if len(inlier_dst_pts) >= 3:
            # Convex hull
            hull = cv2.convexHull(inlier_dst_pts)

            # Rotated bounding box
            rect = cv2.minAreaRect(hull)  # ((cx,cy),(w,h),angle)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            img2_boxed = cv2.polylines(img2.copy(), [box], True, (0,255,0), 3, cv2.LINE_AA)
            plt.figure(figsize=(8,6))
            plt.imshow(img2_boxed, cmap='gray')
            plt.axis('off')
            plt.title("Rotated Bounding Box around Object")
            plt.show()
        else:
            print("Không đủ inliers để vẽ rotated bounding box.")
    else:
        print("Không đủ điểm để tính homography.")

    return matches, kp1, kp2

matches, kp1, kp2 = match_and_draw_box_rotated(r"src\data\sample\sample_1.jpg", r"src\data\sample\3.jpg", max_matches=30)
