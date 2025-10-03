import cv2
import matplotlib.pyplot as plt

def match_images(img1_path, img2_path, max_matches=50):
    # Đọc ảnh
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError("Không thể đọc ảnh. Kiểm tra lại đường dẫn.")

    # Tạo detector ORB
    orb = cv2.ORB_create()

    # Tìm keypoints và descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Dùng Brute-Force matcher (Hamming vì ORB dùng binary descriptor)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Matching
    matches = bf.match(des1, des2)

    # Sắp xếp theo khoảng cách (distance càng nhỏ càng giống)
    matches = sorted(matches, key=lambda x: x.distance)

    # Vẽ kết quả (chỉ lấy max_matches tốt nhất)
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:max_matches], None, flags=2)

    # Hiển thị
    plt.figure(figsize=(12, 6))
    plt.imshow(matched_img)
    plt.axis('off')
    plt.show()

    return matches, kp1, kp2
matches, kp1, kp2 = match_images(r"src\data\sample\1.jpg", r"src\data\sample\sample.jpg", max_matches=30)
