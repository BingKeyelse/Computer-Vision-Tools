import cv2
import numpy as np

import cv2
import numpy as np

def rotate_image_keep_all(img, angle, borderValue=(255,255,255)):
    """Rotate image around center but keep full content (expand canvas)."""
    (h, w) = img.shape[:2]
    angle_rad = np.deg2rad(angle)
    cos = abs(np.cos(angle_rad))
    sin = abs(np.sin(angle_rad))
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # rotation matrix about center then translate
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # adjust translation to move image to center of new canvas
    M[0,2] += (new_w - w) / 2
    M[1,2] += (new_h - h) / 2
    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=borderValue)
    return rotated

def create_detector(detector_name='AKAZE', nfeatures=5000):
    dn = detector_name.lower()
    if dn == 'sift':
        try:
            return cv2.SIFT_create(nfeatures)
        except:
            raise RuntimeError("SIFT not available in your OpenCV build.")
    if dn == 'surf':
        try:
            return cv2.xfeatures2d.SURF_create(400)
        except:
            raise RuntimeError("SURF not available in your OpenCV (needs opencv-contrib).")
    if dn == 'akaze':
        return cv2.AKAZE_create()
    if dn == 'orb':
        return cv2.ORB_create(nfeatures)
    # fallback
    return cv2.AKAZE_create()

def is_binary_descriptor(detector_name):
    return detector_name.lower() in ('akaze', 'orb')

def match_and_find_best_box(template_path, scene_path,
                            detector_name='AKAZE',
                            resize_factor=None,        # None or float scale
                            coarse_step=5,
                            refine_window=6,          # +/- degrees around best coarse
                            refine_step=1,
                            ratio_test=0.75,
                            min_inliers=10,
                            debug=False):
    """
    Returns dict with keys: success(bool), angle, inliers, box (4x2 np.array), scene_result (BGR image)
    """
    # load
    tpl_color = cv2.imread(template_path, cv2.IMREAD_COLOR)
    scene_color = cv2.imread(scene_path, cv2.IMREAD_COLOR)
    if tpl_color is None or scene_color is None:
        raise FileNotFoundError("Cannot read template or scene")

    # optional resize (useful for speed). If None, keep original.
    if resize_factor is not None and resize_factor > 0:
        tpl_color = cv2.resize(tpl_color, (0,0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
        scene_color = cv2.resize(scene_color, (0,0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

    tpl_gray = cv2.cvtColor(tpl_color, cv2.COLOR_BGR2GRAY)
    scene_gray = cv2.cvtColor(scene_color, cv2.COLOR_BGR2GRAY)

    detector = create_detector(detector_name)
    # precompute scene keypoints/descriptors once
    kp_scene, des_scene = detector.detectAndCompute(scene_gray, None)
    if des_scene is None or len(kp_scene) == 0:
        return {'success': False, 'reason': 'no_scene_descriptors'}

    # matcher selection
    if is_binary_descriptor(detector_name):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        # FLANN for float descriptors
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

    best = {'inliers': 0, 'angle': None, 'dst': None, 'M':None, 'tpl_size': None, 'kp_tpl': None, 'good_matches': None}

    # function to process one rotated template
    def process_one(rot_tpl_color):
        rot_tpl_gray = cv2.cvtColor(rot_tpl_color, cv2.COLOR_BGR2GRAY)
        kp_tpl, des_tpl = detector.detectAndCompute(rot_tpl_gray, None)
        if des_tpl is None or len(kp_tpl) == 0:
            return None
        # for FLANN need float32
        if not is_binary_descriptor(detector_name):
            des_tpl_f = np.asarray(des_tpl, np.float32)
            des_scene_f = np.asarray(des_scene, np.float32)
            matches = matcher.knnMatch(des_tpl_f, des_scene_f, k=2)
        else:
            matches = matcher.knnMatch(des_tpl, des_scene, k=2)

        # ratio test
        good = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < ratio_test * n.distance:
                good.append(m)

        if len(good) < 4:
            return {'inliers': 0, 'good':good, 'M':None, 'dst':None, 'kp_tpl':kp_tpl}
        src_pts = np.float32([kp_tpl[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            return {'inliers':0, 'good':good, 'M':None, 'dst':None, 'kp_tpl':kp_tpl}
        h, w = rot_tpl_gray.shape[:2]
        pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        inliers = int(mask.sum()) if mask is not None else 0
        return {'inliers': inliers, 'good': good, 'M':M, 'dst':dst, 'kp_tpl':kp_tpl, 'tpl_size':(w,h)}

    # coarse search
    for angle in range(0, 360, coarse_step):
        rot_tpl = rotate_image_keep_all(tpl_color, angle)
        res = process_one(rot_tpl)
        if res and res['inliers'] > best['inliers']:
            best.update({'inliers': res['inliers'], 'angle': angle, 'dst': res['dst'], 'M': res['M'], 'tpl_size':res.get('tpl_size'), 'kp_tpl':res.get('kp_tpl'), 'good_matches': res.get('good')})
        if debug:
            print(f"[coarse] angle={angle} good={len(res['good']) if res else 0} inliers={res['inliers'] if res else 0}")

    if best['angle'] is None:
        return {'success': False, 'reason': 'no_coarse_match'}

    # refine search around best angle
    start = max(0, best['angle'] - refine_window)
    end = min(359, best['angle'] + refine_window)
    for angle in range(start, end+1, refine_step):
        rot_tpl = rotate_image_keep_all(tpl_color, angle)
        res = process_one(rot_tpl)
        if res and res['inliers'] > best['inliers']:
            best.update({'inliers': res['inliers'], 'angle': angle, 'dst': res['dst'], 'M': res['M'], 'tpl_size':res.get('tpl_size'), 'kp_tpl':res.get('kp_tpl'), 'good_matches': res.get('good')})
        if debug:
            print(f"[refine] angle={angle} good={len(res['good']) if res else 0} inliers={res['inliers'] if res else 0}")

    # final decision
    if best['inliers'] >= min_inliers:
        # draw box on scene_color
        scene_draw = scene_color.copy()
        dst = best['dst']
        cv2.polylines(scene_draw, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)
        cv2.putText(scene_draw, f"angle={best['angle']} inliers={best['inliers']}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        if debug:
            print("BEST:", best['angle'], "inliers:", best['inliers'])
        return {'success': True, 'angle': best['angle'], 'inliers': best['inliers'], 'box': np.int32(dst).reshape(4,2), 'scene_draw': scene_draw, 'M': best['M']}
    else:
        return {'success': False, 'reason': 'not_enough_inliers', 'best_inliers': best['inliers'], 'best_angle': best['angle']}


# Test
result = match_and_find_best_box(
    r"src\data\sample\sample.jpg", 
    r"src\data\sample\2.jpg", 
    detector_name="AKAZE", 
    resize_factor=0.5, 
    debug=True
)

if result['success']:
    cv2.imshow("Result", result['scene_draw'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Fail:", result)
