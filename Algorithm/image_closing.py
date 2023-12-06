import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def image_closing(image_path, kernel_size):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Tạo kernel cho phép đóng ảnh
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Áp dụng phép đóng ảnh
    img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Hiển thị ảnh gốc và ảnh sau phép đóng ảnh
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ảnh Gốc')

    axes[1].imshow(img_closed, cmap='gray')
    axes[1].set_title(f'Phép Đóng Ảnh (kernel_size={kernel_size})')

    plt.show()


image_path = 'input.png'
kernel_size = 50
image_closing(image_path, kernel_size)