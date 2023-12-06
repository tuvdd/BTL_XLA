import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def threshold_segmentation(image_path, threshold_value):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra xem ảnh có tồn tại không
    if img is None:
        print("Không thể đọc được ảnh.")
        return

    # Áp dụng phân đoạn theo ngưỡng
    _, segmented_image = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

    # Hiển thị ảnh gốc và ảnh sau phân đoạn
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ảnh Gốc')

    axes[1].imshow(segmented_image, cmap='gray')
    axes[1].set_title(f'Phân đoạn theo ngưỡng {threshold_value}')

    plt.show()


image_path = 'input.png'
threshold_value = 100  # Thay đổi giá trị ngưỡng tùy ý
threshold_segmentation(image_path, threshold_value)