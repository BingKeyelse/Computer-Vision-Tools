import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def template_match_demo(template_path, target_path, method=cv2.TM_CCOEFF_NORMED):
    """
    1. Đọc template và target (grayscale)
    2. Thực hiện template matching
    3. Vẽ bounding box kết quả
    4. Trả về time processing
    """
    # Đọc ảnh grayscale
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

    if template is None or target is None:
        raise ValueError("Không thể đọc ảnh. Kiểm tra lại đường dẫn.")

    start_time = time.time()
    
    # Template matching
    res = cv2.matchTemplate(target, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # Tùy method, max hoặc min là best match
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    h, w = template.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Vẽ bounding box
    target_boxed = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(target_boxed, top_left, bottom_right, (0,255,0), 3)

    end_time = time.time()
    processing_time = end_time - start_time
    print(processing_time)
    

    # Hiển thị
    plt.figure(figsize=(8,6))
    plt.imshow(target_boxed[..., ::-1])  # chuyển BGR -> RGB cho matplotlib
    plt.axis('off')
    plt.title(f"Template Matching - Time: {processing_time:.4f}s")
    plt.show()
    

    # return processing_time

template_match_demo(r"src\data\sample\sample.jpg", r"src\data\sample\2.jpg")
