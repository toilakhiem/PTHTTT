import tensorflow as tf
import numpy as np

# Tải dữ liệu MNIST - Code tải dữ liệu hình ảnh chứa các chữ số viết tay và nhãn tương ứng cho từng hình ảnh từ bộ dữ liệu MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Chuẩn hóa dữ liệu
x_train, x_test = x_train / 255.0, x_test / 255.0

# Xây dựng mô hình mạng nơ-ron
# Một mô hình mạng nơ-ron được xây dựng với kiến trúc gồm các lớp,
# bao gồm lớp Flatten (để biến đổi hình ảnh thành vectơ),
# lớp Dense (mạng nơ-ron fully connected) với 128 đơn vị nơ-ron và hàm kích hoạt ReLU,
# lớp Dropout để tránh overfitting,
# và lớp Dense cuối cùng với 10 đơn vị nơ-ron tương ứng với 10 lớp đầu ra cho việc phân loại các chữ số từ 0 đến 9.
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# Dự đoán với một mẫu dữ liệu
predictions = model(x_train[:1])

# Định nghĩa hàm mất mát
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions)

# Compile mô hình
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(x_train, y_train, epochs=5)

# Đánh giá mô hình và in ra kết quả
model.evaluate(x_test, y_test, verbose=2)
