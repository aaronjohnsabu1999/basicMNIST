from tensorflow.keras.utils    import to_categorical
from tensorflow.keras.layers   import Dense
from tensorflow.keras.models   import Sequential
from tensorflow.keras.datasets import mnist
import numpy             as np
import tensorflow        as tf
import matplotlib.pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

print('Using Tensorflow version ', tf.__version__)
print()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train_encoded = to_categorical(y_train)
y_test_encoded  = to_categorical(y_test)

print('x_train         shape: ', x_train.shape)
print('y_train         shape: ', y_train.shape)
print('y_train_encoded shape: ', y_train_encoded.shape)
print('x_test          shape: ', x_test.shape)
print('y_test          shape: ', y_test.shape)
print('y_test_encoded  shape: ', y_test_encoded.shape)

# for i in range(10):
#     plt.title(y_train[i])
#     plt.imshow(x_train[i], cmap = 'binary')
#     plt.show()

x_train_reshaped = np.reshape(x_train, (60000, 784))
x_test_reshaped  = np.reshape(x_test,  (10000, 784))

epsilon = 1e-10
x_mean  = np.mean(x_train_reshaped)
x_std   = np.std(x_train_reshaped)
x_train_norm = (x_train_reshaped - x_mean)/(x_std + epsilon)
x_test_norm  = (x_test_reshaped  - x_mean)/(x_std + epsilon)

model = Sequential([
    Dense(128, activation = 'relu', input_shape = (784, )),
    Dense(128, activation = 'relu'),
    Dense( 10, activation = 'softmax')
])
model.compile(
    optimizer = 'sgd',
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)
model.summary()

model.fit(x_train_norm, y_train_encoded, epochs = 5)
model.evaluate(x_test_norm, y_test_encoded)

preds = model.predict(x_test_norm)
print('Shape of preds:', preds.shape)
start_index = 0
plt.figure(figsize = (12, 12))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    pred = np.argmax(preds[start_index + i])
    gt   = y_test[start_index + i]

    col  = 'g'
    if pred != gt:
        col = 'r'

    plt.xlabel('i = {}, pred = {}, gt = {}'.format(start_index + i, pred, gt))
    plt.imshow(x_test[start_index + i], cmap = 'binary')
plt.show()

plt.plot(preds[8])
plt.show()