import cv2
import numpy as np

def rotate_image(image, angle):
    """Hàm để xoay một hình ảnh theo một góc nhất định mà vẫn giữ nguyên hình dạng của vật"""
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Tính toán kích thước mới của ảnh
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    
    # Điều chỉnh ma trận xoay để đảm bảo ảnh không bị cắt
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Xoay ảnh với kích thước mới
    rotated = cv2.warpAffine(image, M, (new_w, new_h))
    return rotated

def match_and_draw_rotations(template_path, scene_path):
    # Load ảnh
    template_ori = cv2.imread(template_path, cv2.IMREAD_COLOR)
    scene = cv2.imread(scene_path, cv2.IMREAD_COLOR)

    # Resize nhỏ để chạy nhanh hơn
    template_ori = cv2.resize(template_ori, (template_ori.shape[1]//2, template_ori.shape[0]//2))
    scene = cv2.resize(scene, (scene.shape[1]//2, scene.shape[0]//2))

    # ORB detector
    orb = cv2.ORB_create(5000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    best_matches = []
    best_dst = None
    best_angle = None
    best_kp1, best_kp2 = None, None
    best_good = None

    # Quét góc xoay
    for angle in range(0, 360, 1):
        template = rotate_image(template_ori, angle)

        kp1, des1 = orb.detectAndCompute(template, None)
        kp2, des2 = orb.detectAndCompute(scene, None)
        if des1 is None or des2 is None:
            continue

        matches = bf.knnMatch(des1, des2, k=2)

        # Ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) > 5:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                h, w = template.shape[:2]
                pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts, M)

                

                if len(good) > len(best_matches):
                    best_matches = good
                    best_dst = dst
                    best_angle = angle
                    best_kp1, best_kp2 = kp1, kp2
                    best_good = good

    # Vẽ kết quả
    if best_dst is not None:
        scene_result = scene.copy()
        # Vẽ bounding box
        cv2.polylines(scene_result, [np.int32(best_dst)], True, (0, 255, 0), 3)
        cv2.putText(scene_result, f"Angle: {best_angle} deg", (30,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Vẽ keypoint matches (chỉ lấy 30 cái để dễ nhìn)
        match_img = cv2.drawMatches(template_ori, best_kp1, scene_result, best_kp2,
                                    best_good[:30], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Vẽ bounding box với số thứ tự góc
        for i, p in enumerate(dst):
            x, y = int(p[0][0]), int(p[0][1])
            cv2.circle(scene_result, (x,y), 5, (255,0,0), -1)
            cv2.putText(scene_result, str(i+1), (x+10,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        cv2.imshow("Best Match with Box", match_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Không tìm thấy match nào đủ tốt.")

# Test
match_and_draw_rotations(r"src\data\sample\sample.jpg", r"src\data\sample\2.jpg")
