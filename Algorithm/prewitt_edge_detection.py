import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def prewitt_edge_detection(image_path):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Áp dụng bộ lọc Prewitt theo hướng ngang và hướng dọc
    prewitt_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    prewitt_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Tính toán biên bằng cách kết hợp biên từ cả hai hướng
    prewitt_edges = np.sqrt(prewitt_x**2 + prewitt_y**2)

    # Hiển thị ảnh gốc và ảnh sau khi áp dụng Kỹ thuật Prewitt
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ảnh Gốc')

    axes[1].imshow(prewitt_edges, cmap='gray')
    axes[1].set_title('Ảnh Sau Kỹ thuật Prewitt')

    plt.show()


image_path = 'input.png'
prewitt_edge_detection(image_path)