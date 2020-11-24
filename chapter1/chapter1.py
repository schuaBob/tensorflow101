import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mping
model = keras.Sequential(name='Sequential')
model.add(layers.Dense(64, activation='relu', input_shape=(784,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
# model = tf.keras.Sequential([
#     layers.Dense(64, activation='relu', input_shape=(784,)),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])
# cmd (admin privilege)
# dot -v
# dot -c
# dot -v
plot_model(model, to_file='Sequential_model.png', show_shapes=True, show_layer_names=True)
img = mping.imread('Sequential_model.png')
plt.imshow(img)
plt.show()
