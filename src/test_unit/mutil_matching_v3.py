import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=borderValue)

def match_and_draw_box_rotated(template_path, scene_path, use_akaze=True,
                               coarse_step=10, refine_window=5, refine_step=1,
                               ratio_test=0.75, min_inliers=8, resize_factor=0.5, debug=True):
    # Đọc ảnh
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    scene = cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)
    if template is None or scene is None:
        raise FileNotFoundError("Không đọc được ảnh template hoặc scene")

    # Resize cho nhanh
    if resize_factor:
        template = cv2.resize(template, (0,0), fx=resize_factor, fy=resize_factor)
        scene = cv2.resize(scene, (0,0), fx=resize_factor, fy=resize_factor)

    # Detector
    detector = cv2.AKAZE_create() if use_akaze else cv2.SIFT_create()
    kp2, des2 = detector.detectAndCompute(scene, None)
    if des2 is None:
        print("Không có đặc trưng trong scene.")
        return

    # Matcher
    if use_akaze:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

    best = {'inliers': 0, 'angle': None, 'M': None, 'dst': None, 'good': []}

    def process_one(rot_tpl):
        kp1, des1 = detector.detectAndCompute(rot_tpl, None)
        if des1 is None:
            return None

        matches = matcher.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < ratio_test * n.distance:
                good.append(m)
        if len(good) < 3:
            return None

        # Dùng affine transform thay vì homography
        good = sorted(good, key=lambda x: x.distance)[:20]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

        if M is None or inliers is None:
            return None

        inliers_count = int(inliers.sum())
        h, w = rot_tpl.shape[:2]
        pts = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        dst = cv2.transform(pts, M)
        return {'inliers': inliers_count, 'M': M, 'dst': dst, 'good': good, 'kp1': kp1}

    # coarse search
    for angle in range(0, 360, coarse_step):
        rot = rotate_image_keep_all(template, angle)
        res = process_one(rot)
        if res and res['inliers'] > best['inliers']:
            best.update(res)
            best['angle'] = angle
        if debug:
            print(f"[coarse] angle={angle} inliers={res['inliers'] if res else 0}")

    if best['angle'] is None:
        print("Không tìm thấy góc phù hợp (coarse search).")
        return

    # refine search
    start = max(0, best['angle'] - refine_window)
    end = min(359, best['angle'] + refine_window)
    for angle in range(start, end+1, refine_step):
        rot = rotate_image_keep_all(template, angle)
        res = process_one(rot)
        if res and res['inliers'] > best['inliers']:
            best.update(res)
            best['angle'] = angle
        if debug:
            print(f"[refine] angle={angle} inliers={res['inliers'] if res else 0}")

    if best['inliers'] < min_inliers:
        print("Không đủ inliers để xác định affine transform.")
        return

    # Hiển thị kết quả
    print(f"\n✅ Best angle: {best['angle']}°, Inliers: {best['inliers']}")
    print("Affine matrix:\n", best['M'])

    # Warp template sang scene (theo affine)
    aligned = cv2.warpAffine(template, best['M'], (scene.shape[1], scene.shape[0]))
    cv2.imshow("Warped Template", aligned)

    # Vẽ box trên ảnh scene
    scene_color = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    cv2.polylines(scene_color, [np.int32(best['dst'])], True, (0,255,0), 3)

    # Hiển thị match
    matched = cv2.drawMatches(template, best['kp1'], scene_color, kp2, best['good'], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(14,7))
    plt.imshow(matched[...,::-1])
    plt.axis("off")
    plt.title(f"Best angle: {best['angle']}°, Inliers: {best['inliers']}")
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Test ---
match_and_draw_box_rotated(
    r"src\data\sample\sample.jpg",
    r"src\data\sample\5.jpg",
    use_akaze=False,
    coarse_step=10,
    refine_window=5,
    refine_step=1,
    resize_factor=0.5,
    debug=True
)
