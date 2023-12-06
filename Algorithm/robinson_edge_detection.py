import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def robinson_edge_detection(image_path):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Bộ lọc Robinson theo 8 hướng khác nhau
    robinson_filters = [
        np.array([1, 2, 1, 0, 0, 0, -1, -2, -1]),
        np.array([2, 1, 0, 1, 0, -1, 0, -1, -2]),
        np.array([1, 0, -1, 2, 0, -2, 1, 0, -1]),
        np.array([0, -1, -2, 1, 0, -1, 2, 1, 0]),
        np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1]),
        np.array([-2, -1, 0, -1, 0, 1, 0, 1, 2]),
        np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]),
        np.array([0, 1, 2, -1, 0, 1, -2, -1, 0])
    ]

    # Áp dụng bộ lọc Robinson
    robinson_edges = np.zeros_like(img, dtype=np.float32)
    for kernel in robinson_filters:
        kernel = kernel.reshape((3, 3))
        filtered = cv2.filter2D(img, cv2.CV_64F, kernel)
        robinson_edges = np.maximum(robinson_edges, filtered)

    # Chuyển đổi giá trị về 8-bit unsigned integer
    robinson_edges = np.uint8(np.abs(robinson_edges))

    # Hiển thị ảnh gốc và ảnh sau khi áp dụng Kỹ thuật Robinson
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ảnh Gốc')

    axes[1].imshow(robinson_edges, cmap='gray')
    axes[1].set_title('Ảnh Sau Kỹ thuật Robinson')

    plt.show()


image_path = 'input.png'
robinson_edge_detection(image_path)