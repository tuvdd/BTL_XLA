import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def fill_region(image_path, seed_point):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Sao chép ảnh để tránh ảnh gốc bị thay đổi
    filled_image = img.copy()

    # Ngưỡng đen trắng để tạo mask
    _, mask = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Tạo một hình ảnh với tất cả giá trị là 0
    height, width = img.shape
    filled = np.zeros((height + 2, width + 2), dtype=np.uint8)

    # Điền đầy vùng từ điểm hạt giống (seed point)
    cv2.floodFill(filled_image, filled, seed_point, 255)

    # Hiển thị ảnh gốc và ảnh sau khi điền đầy vùng
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ảnh Gốc')

    axes[1].imshow(filled_image, cmap='gray')
    axes[1].set_title('Điền Đầy Vùng')

    plt.show()


image_path = 'input.png'
seed_point = (50, 60)

fill_region(image_path, seed_point)