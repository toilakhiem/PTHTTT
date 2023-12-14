from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load dữ liệu từ bộ CIFAR-10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Chuẩn hóa giá trị pixel của hình ảnh để nằm trong khoảng từ 0 đến 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Định nghĩa tên các lớp tương ứng trong bộ dữ liệu CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Tạo một đối tượng figure để hiển thị hình ảnh
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])  # Xóa các ghi chú trục x
    plt.yticks([])  # Xóa các ghi chú trục y
    plt.grid(False)  # Tắt lưới
    plt.imshow(train_images[i])  # Hiển thị hình ảnh thứ i
    plt.xlabel(class_names[train_labels[i][0]])  # Đặt nhãn dưới hình ảnh
plt.show()
