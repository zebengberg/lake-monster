import os
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(2,)))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Dense(1))

print(model.summary())
model.compile('adam', 'mse', ['mae'])


def build_xy(n):
  x = np.random.random((n, 2))
  x2 = x ** 2
  y = x2.sum(axis=1)
  return x, y


n = int(1e6)
x, y = build_xy(n)
model.fit(x, y, 64, 2)

n = int(1e5)
x, y = build_xy(n)
model.evaluate(x, y)

tfjs.converters.save_keras_model(model, os.path.dirname(__file__) + '/model')
