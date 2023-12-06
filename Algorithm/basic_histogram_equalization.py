import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
def basic_histogram_equalization(image_path):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Biến đổi âm bản
    equalized_img = cv2.equalizeHist(img)

    # Hiển thị ảnh gốc và ảnh đã được biến đổi âm bản
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ảnh Gốc')

    axes[1].imshow(equalized_img, cmap='gray')
    axes[1].set_title('Biến Đổi Âm Bản')

    plt.show()

image_path = 'input.png'
basic_histogram_equalization(image_path)