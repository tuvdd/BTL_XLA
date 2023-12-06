import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def smoothing_and_sharpening(image_path):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Làm mịn ảnh
    smoothed_img = cv2.GaussianBlur(img, (5, 5), 0)

    # Làm nét ảnh
    sharpened_img = cv2.filter2D(img, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

    # Hiển thị ảnh gốc, ảnh đã làm mịn và ảnh đã làm nét
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ảnh Gốc')

    axes[1].imshow(smoothed_img, cmap='gray')
    axes[1].set_title('Ảnh Làm Mịn')

    axes[2].imshow(sharpened_img, cmap='gray')
    axes[2].set_title('Ảnh làm nét')

    plt.show()


image_path = 'input.png'
smoothing_and_sharpening(image_path)