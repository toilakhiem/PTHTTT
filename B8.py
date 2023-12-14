from keras.datasets import imdb
from keras import preprocessing
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Flatten, Dense
import matplotlib.pyplot as plt

# Đặt số lượng features tối đa và độ dài tối đa của một sequence
max_features = 10000
maxlen = 20

# Tải bộ dữ liệu IMDB và giới hạn số lượng từ tối đa
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Chuẩn hóa độ dài của các sequences
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# Xây dựng mô hình
model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Biên dịch mô hình
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# Hiển thị tóm tắt mô hình
model.summary()

# Huấn luyện mô hình
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Vẽ đồ thị độ chính xác
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Vẽ đồ thị mất mát
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
