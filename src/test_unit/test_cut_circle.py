import cv2
import numpy as np

# --- 1. Đọc ảnh ---
img_path = r"D:\Desktop_with_Data_Pronics\Project_Duy_Nguyen\Duy-Nguyen-Project (copy)\picture\saved_sample.png"
template = cv2.imread(img_path)

# --- 2. Tạo mask tròn ---
h, w = template.shape[:2]
center = (w // 2, h // 2)
radius = min(center) - 5

mask = np.zeros_like(template, dtype=np.uint8)
cv2.circle(mask, center, radius, (255, 255, 255), -1)

# --- 3. Cắt vùng tròn ---
circle_only = cv2.bitwise_and(template, mask)

# --- 4. Hàm xoay ảnh ---
def rotate_image(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return rotated

# --- 5. Xoay 360 độ ---
for angle in range(0, 360, 10):  # mỗi lần xoay 30 độ
    rotated = rotate_image(template, angle)
    display = rotated.copy()
    cv2.putText(display, f"{angle} deg", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("Rotating Circle", display)
    cv2.imshow(" Circle", mask)
    cv2.waitKey(0)  # dừng 0.3 giây giữa các khung

cv2.destroyAllWindows()
