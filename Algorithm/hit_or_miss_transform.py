import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def hit_or_miss_transform(image_path, kernel):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Áp dụng phép biến đổi Hit-or-Miss
    result = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)

    # Hiển thị ảnh gốc và ảnh sau phép biến đổi Hit-or-Miss
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ảnh Gốc')

    axes[1].imshow(result, cmap='gray')
    axes[1].set_title('Kết Quả Phép Hit-or-Miss')

    plt.show()


image_path = 'input.png'

# Định nghĩa kernel cho phép biến đổi Hit-or-Miss
kernel = np.array([[0, 1, 0],
                   [-1, 1, 1],
                   [0, 1, 0]], dtype=np.int8)

# Gọi hàm để thực hiện phép biến đổi Hit-or-Miss
hit_or_miss_transform(image_path, kernel)