import cv2
import numpy as np
import matplotlib.pyplot as plt

def match_and_draw_box(template_path, scene_path, use_akaze=True):
    # Đọc ảnh
    img1 = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)

    # Resize nhỏ 1/2 cho dễ hiển thị
    img1 = cv2.resize(img1, (img1.shape[1]//2, img1.shape[0]//2))
    img2 = cv2.resize(img2, (img2.shape[1]//2, img2.shape[0]//2))

    # Chọn detector
    if use_akaze:
        detector = cv2.AKAZE_create()
    else:
        detector = cv2.SIFT_create()

    # Tìm keypoint và descriptor
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        print("Không tìm thấy đặc trưng nào.")
        return

    # Matching
    if use_akaze:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
    else:
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
    print(len(good))

    if len(good) >= 5:
        # Lấy 15 match tốt nhất
        good = sorted(good, key=lambda x: x.distance)[:15]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        # Tính affine transform (xoay + dịch + scale)
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

        if M is not None:
            # Lấy rotation + dịch
            R = M[:2,:2]
            t = M[:,2]

            # Góc xoay
            angle = np.degrees(np.arctan2(R[1,0], R[0,0]))
            print("Rotation matrix:\n", R)
            print("Translation:", t)
            print("Rotation angle:", angle)

            # Vẽ bounding box
            h,w = img1.shape
            pts = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
            dst_box = cv2.transform(pts, M)

            img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            cv2.polylines(img2_color, [np.int32(dst_box)], True, (0,255,0), 3)

            # Vẽ matches để debug
            matched_img = cv2.drawMatches(img1, kp1, img2_color, kp2, good, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            plt.figure(figsize=(14,7))
            plt.imshow(matched_img[...,::-1])  # BGR -> RGB
            plt.axis("off")
            plt.show()
        else:
            print("Không tính được affine transform")
    else:
        print("Không đủ matches")

# Test
match_and_draw_box( r"src\data\sample\sample_1.jpg",r"src\data\sample\4.jpg", use_akaze=False)
