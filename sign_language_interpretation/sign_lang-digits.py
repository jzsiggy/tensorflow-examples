from resources import get_image

import tensorflow as tf
import numpy as np
from numpy import shape
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

x = np.load("sign_language_interpretation/sign-language-digits-dataset/X.npy")
y = np.load("sign_language_interpretation/sign-language-digits-dataset/Y.npy")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=RANDOM_SEED)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(64, 64)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)


# print(model.predict(x_test[3].reshape(1, 64, 64)))


input_image = get_image()
print(model.predict(input_image.reshape(1, 64, 64)))

# plt.imshow(input_image)
# plt.show()


