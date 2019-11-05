import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')


xs = np.arange(1, 15);
ys = xs ** 2

# xs = np.array(xs, dtype=float)
# ys = np.array(ys, dtype=float)

# xs = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
# ys = np.array([1, 4, 9, 16, 25, 36, 49, 64], dtype=float)

# print(ys)


model.fit(xs, ys, epochs=150)

print(model.predict([20]))