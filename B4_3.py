import tensorflow as tf
import matplotlib.pyplot as plt

# In ra phiên bản của TensorFlow
print(tf.__version__)

# Tải bộ dữ liệu Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Định nghĩa tên lớp cho các nhãn trong bộ dữ liệu
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Kiểm tra hình dạng của tập dữ liệu huấn luyện
print(train_images.shape)

# Hiển thị hình ảnh đầu tiên từ tập huấn luyện
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()  # Thêm thanh chỉ báo màu
plt.grid(False)  # Tắt lưới
plt.show()

# Chuẩn hóa các giá trị pixel của hình ảnh từ 0-255 về phạm vi 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Vẽ và hiển thị 25 hình ảnh đầu tiên từ tập huấn luyện
plt.figure(figsize=(10,10))
for i in range(30):
    plt.subplot(5, 6, i+1)
    plt.xticks([])  # Xóa các ghi chú trục x
    plt.yticks([])  # Xóa các ghi chú trục y
    plt.grid(False)  # Tắt lưới
    plt.imshow(train_images[i], cmap=plt.cm.binary)  # Sử dụng colormap binary
    plt.xlabel(class_names[train_labels[i]])
plt.show()
