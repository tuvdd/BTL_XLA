import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def apply_log_transform(image_path, c=1):
    # Đọc ảnh
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Áp dụng biến đổi logarith
    log_transformed = c * np.log1p(image)

    # Chuyển đổi kiểu dữ liệu về uint8 để có thể hiển thị bằng matplotlib
    log_transformed = np.uint8(log_transformed)

    # Hiển thị ảnh gốc và ảnh sau biến đổi logarith
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Ảnh Gốc')

    axes[1].imshow(log_transformed, cmap='gray')
    axes[1].set_title('Biến Đổi Logarith')

    plt.show()


# Đường dẫn đến ảnh
image_path = 'input.png'
c_value = 2

# Áp dụng biến đổi logarith
apply_log_transform(image_path, c_value)