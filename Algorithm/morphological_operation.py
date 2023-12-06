import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def morphological_operation(image_path):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Áp dụng phép co (dilation) để làm nổi bật các cạnh
    kernel = np.ones((5, 5), np.uint8)  # Kích thước kernel
    dilated_img = cv2.dilate(img, kernel, iterations=1)

    # Hiển thị ảnh gốc và ảnh sau phép co
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ảnh Gốc')

    axes[1].imshow(dilated_img, cmap='gray')
    axes[1].set_title('Ảnh Sau Phép Co')

    plt.show()


image_path = 'input.png'
morphological_operation(image_path)