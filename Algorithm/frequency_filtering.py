import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def frequency_filtering(image_path, cutoff_frequency):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Thực hiện biến đổi Fourier
    f_transform = np.fft.fft2(img)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Tạo bộ lọc thông thấp
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2  # Tìm tâm của hình ảnh
    mask = np.ones((rows, cols), np.uint8)
    r = cutoff_frequency
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0

    # Áp dụng bộ lọc
    f_transform_shifted = f_transform_shifted * mask

    # Thực hiện ngược biến đổi Fourier
    f_ishift = np.fft.ifftshift(f_transform_shifted)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)

    # Hiển thị ảnh gốc và ảnh đã được lọc
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ảnh Gốc')

    axes[1].imshow(img_filtered, cmap='gray')
    axes[1].set_title(f'Bộ Lọc Tần Số (cutoff={cutoff_frequency})')

    plt.show()


image_path = 'input.png'
cutoff_frequency = 20
frequency_filtering(image_path, cutoff_frequency)