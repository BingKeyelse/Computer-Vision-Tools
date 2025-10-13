import cv2
import numpy as np

def extract_oriented_object(img, start, end, angle_rad):
    """
    Chuẩn hóa vùng oriented box về phương nằm ngang.
    - start, end: 2 điểm đối diện nhau (đường chéo)
    - angle_rad: góc nghiêng (radian)
    """
    # Chuyển góc sang độ
    angle_deg = np.degrees(angle_rad)

    # Tâm và kích thước box
    cx, cy = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    w = abs(end[0] - start[0])
    h = abs(end[1] - start[1])

    # Tạo rotated rect (giống cv2.minAreaRect)
    rect = ((cx, cy), (w, h), angle_deg)

    # Lấy 4 điểm polygon từ rect (theo hướng nghiêng)
    box = cv2.boxPoints(rect).astype(np.float32)

    # --- Tạo ma trận xoay để "dựng thẳng" box ---
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Tính kích thước mới sau khi xoay toàn ảnh
    h_img, w_img = img.shape[:2]
    new_w = int(h_img * sin + w_img * cos)
    new_h = int(h_img * cos + w_img * sin)

    # Cập nhật offset để không bị crop ngoài
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    # Xoay toàn ảnh để đối tượng về thẳng
    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    # Chuyển box điểm sang không gian mới (sau khi xoay)
    ones = np.ones((4, 1), dtype=np.float32)
    box_h = np.hstack([box, ones])
    box_rotated = (M @ box_h.T).T

    # Crop vùng bounding box mới
    x_min, y_min = box_rotated[:, 0].min(), box_rotated[:, 1].min()
    x_max, y_max = box_rotated[:, 0].max(), box_rotated[:, 1].max()

    cropped = rotated[int(y_min):int(y_max), int(x_min):int(x_max)]
    return cropped, rotated, box_rotated.astype(np.int32)

def test_oriented_box(data_list):
    for item in data_list:
        link = item["link"]
        shape_type, start, end, angle = item["data"]

        img = cv2.imread(link)
        if img is None:
            print(f"❌ Không đọc được ảnh: {link}")
            continue

        cropped, rotated, box_rotated = extract_oriented_object(img, start, end, angle)

        # Vẽ vùng oriented box sau xoay
        vis = rotated.copy()
        cv2.polylines(vis, [box_rotated], isClosed=True, color=(0, 255, 0), thickness=2)
        vis= cv2.resize(vis, (0,0), fx= 0.5, fy=0.5)
        cv2.imshow("Rotated Scene", vis)
        cv2.imshow("Aligned Crop", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# --- Test ---
data_list = [
    {
        'mode': 0,
        'link': 'D:/Desktop_with_Data_Pronics/Computer_Vision_Tool/src/data/sample/mutil/5.jpg',
        'data': (
            'oriented_box',
            (1109.6116688897837, 1396.8098268567962),
            (1367.7384198744082, 1535.2759316957988),
            0.37082144990784993  # radian
        )
    }
]

test_oriented_box(data_list)
