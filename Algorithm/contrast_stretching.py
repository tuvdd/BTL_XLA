import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def contrast_stretching(image_path, min_output, max_output):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Normalization
    img_normalized = cv2.normalize(img, None, min_output, max_output, cv2.NORM_MINMAX)

    # Hiển thị ảnh gốc và ảnh sau khi biến đổi giãn ảnh
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ảnh Gốc')

    axes[1].imshow(img_normalized, cmap='gray')
    axes[1].set_title('Biến Đổi Giãn Ảnh')

    plt.show()


image_path = 'input.png'
min_output = 0
max_output = 255
contrast_stretching(image_path, min_output, max_output)