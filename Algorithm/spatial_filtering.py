import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def spatial_filtering(image_path, kernel):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Áp dụng bộ lọc không gian
    filtered_img = cv2.filter2D(img, -1, kernel)

    # Hiển thị ảnh gốc và ảnh đã được xử lý
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ảnh Gốc')

    axes[1].imshow(filtered_img, cmap='gray')
    axes[1].set_title('Bộ lọc không gian')

    plt.show()


image_path = 'input.png'

kernel = np.ones((3, 3), np.float32) / 9.0
kernel1 = np.ones((6, 6), np.float32) / 12.0

spatial_filtering(image_path, kernel)
spatial_filtering(image_path, kernel1)