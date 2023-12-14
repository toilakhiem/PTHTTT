from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
# Tải bộ dữ liệu Pima Indians Diabetes
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# Tách bộ dữ liệu thành input (X) và output (Y)
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Tạo mô hình
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))  # Lớp ẩn với 12 nút
model.add(Dense(8, activation='relu'))  # Lớp ẩn thứ hai với 8 nút
model.add(Dense(1, activation='sigmoid'))  # Lớp đầu ra

# Biên dịch mô hình
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)

# In ra các khóa dữ liệu trong lịch sử
print(history.history.keys())

# Tóm tắt lịch sử cho độ chính xác
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Tóm tắt lịch sử cho mất mát
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
