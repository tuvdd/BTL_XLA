import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def log_transform(image_path, c=1):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Normalize giá trị pixel về đoạn [0, 1]
    normalized_img = img / 255.0

    # Áp dụng phép biến đổi hàm log
    log_transformed_img = c * np.log1p(normalized_img)

    # Scale lại giá trị pixel về đoạn [0, 255]
    log_transformed_img = (log_transformed_img * 255).astype(np.uint8)

    # Hiển thị ảnh gốc và ảnh đã được biến đổi
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ảnh Gốc')

    axes[1].imshow(log_transformed_img, cmap='gray')
    axes[1].set_title('Biến Đổi Hàm Log')

    plt.show()

image_path = 'input.png'
log_transform(image_path, c=1)