import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

train_data, info = tfds.load('cifar10', split = 'train[10%:]', with_info=True)
valid_data = tfds.load('cifar10', split='train[:10%]')
test_data = tfds.load('cifar10', split='test')

labels_dict = dict(enumerate(info.features['label'].names))
train_dict = {}
for data in train_data:
    label = data['label'].numpy()
    train_dict[label] = train_dict.setdefault(label, 0) + 1

def flip(x):
    x = tf.image.random_flip_left_right(x)
    return x

def color(x):
    x = tf.image.random_hue(x, 0.08) # 隨機調整影像色調
    x = tf.image.random_saturation(x, 0,6, 1.6) # 隨機影像飽和度
    x = tf.image.random_brightness(x, 0.05) # 隨機影像亮度
    x = tf.image.random_contrast(x, 0.7, 1.3) # 隨機影像對比度
    return x

def rotate(x):
    # 隨機旋轉n次
    x = tf.image.rot90(x, tf.random.uniform(shape=[], minval= 1, maxval=4, dtype=tf.int32))
    return x

def zoom(x, scale_min=0.6, scale_max=1.4):
    h, w, c = x.shape
    scale = tf.random.uniform([], scale_min, scale_max)
    sh = h*scale
    sw = w*scale
    x = tf.image.resize(x, (sh, sw)) #  縮放
    x = tf.image.resize_with_crop_or_pad(x, h, w) # 縮放後的填補
    return x

def parse_aug_fn(dataset):
    # 像素縮小255倍, 變成0-1之間
    x = tf.cast(dataset['image'], tf.float32) / 255
    # 觸發水平翻轉
    x = flip(x)
    # 50%觸發顏色轉換
    x = tf.cond(tf.random.uniform(shape=[], 0, 1) > 0.5, lambda: color(x), lambda: x)
    # 25%觸發旋轉
    x = tf.cond(tf.random.uniform(shape=[], 0, 1) > 0.75, lambda: rotate(x), lambda: x)
    # 50%觸發縮放
    x = tf.cond(tf.random.uniform(shape=[], 0, 1) > 0.5, lambda: zoom(x), lambda: x)
    y = tf.one_hot(dataset['label'], 10)
    return x, y


def parse_fn(dataset):
    # 像素縮小255倍, 變成0-1之間
    x = tf.cast(dataset['image'], tf.float32) / 255
    y = tf.one_hot(dataset['label'], 10)
    return x, y

AUTOTUNE = tf.data.experimental.AUTOTUNE
batch_size = 1
train_num = int(info.splits['train'].num_examples / 10) * 9

train_data = train_data.shuffle(train_num)
# 載入預處理, cpu自動調整
train_data = train_data.map(map_func=parse_aug_fn, num_parallel_calls=AUTOTUNE)
# 批次64, prefetch開啟,暫存空間自動調整
train_data = train_data.batch(batch_size).prefetch(buffer_size = AUTOTUNE)
valid_data = valid_data.map(map_func=parse_fn, num_parallel_calls=AUTOTUNE)
valid_data = valid_data.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
test_data = test_data.map(map_func=parse_fn, num_parallel_calls=AUTOTUNE)
test_data = test_data.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

# build model
model_name = 'convolution_model_with_aug_data'
inputs = keras.Input(shape=(32,32,3))
x = layers.Conv2D(64, (3,3), activation='relu')(inputs)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.Conv2D(256, (3, 3), activation='relu')(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs = outputs, name=model_name)
model.summary()
BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(BASEDIR,'models')
log_dir = os.path.join(BASEDIR, 'logs', model_name)
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
model_mckp = keras.callbacks.ModelCheckpoint(
    os.path.join(model_dir,f'{model_name}.h5'),
    monitor='val_categorical_accuracy',
    save_best_only=True,
    mode='max'
)
model.compile(
    keras.optimizers.Adam(),
    loss = keras.losses.CategoricalCrossentropy(),
    metrics = [keras.metrics.CategoricalAccuracy()]
)
model.fit(train_data, epochs=100, validation_data=valid_data, callbacks=[model_cbk, model_mckp])
