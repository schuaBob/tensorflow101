from .preprocessing import parse_fn, parse_aug_fn
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras import initializers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def build_and_train_model(run_name):
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(64, (3, 3))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, (3, 3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=run_name)
    log_files = os.path.join(BASEDIR, 'logs', run_name)
    # histogram_freq = 1, 每次epoch各層的網路權重分布都會被記錄下來
    model_cbk = keras.callbacks.TensorBoard(log_dir=log_files)
    modelfiles = os.path.join(BASEDIR, 'models', f'{run_name}.h5')
    model_mckp = keras.callbacks.ModelCheckpoint(
        modelfiles,
        monitor='val_categorical_accuracy',
        save_best_only=True,
        mode='max'
    )
    # 設定
    model.compile(
        keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()]
    )
    model.fit(train_data, epochs=100, validation_data=valid_data,
              callbacks=[model_cbk, model_mckp])


train_data, info = tfds.load('cifar10', split='train[10%:]', with_info=True)
valid_data = tfds.load('cifar10', split='train[:10%]')
test_data = tfds.load('cifar10', split='test')
AUTOTUNE = tf.data.experimental.AUTOTUNE
batch_size = 1
train_num = int(info.splits['train'].num_examples / 10) * 9

train_data = train_data.shuffle(train_num)
# 載入預處理, cpu自動調整
train_data = train_data.map(map_func=parse_aug_fn, num_parallel_calls=AUTOTUNE)
# 批次64, prefetch開啟,暫存空間自動調整
train_data = train_data.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
valid_data = valid_data.map(map_func=parse_fn, num_parallel_calls=AUTOTUNE)
valid_data = valid_data.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
test_data = test_data.map(map_func=parse_fn, num_parallel_calls=AUTOTUNE)
test_data = test_data.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

model_name='model_with_batchnormalization'
build_and_train_model(run_name=model_name)
model = keras.models.load_model(os.path.join(BASEDIR, 'models', f'{model_name}.h5'))
loss, acc = model.evaluate(test_data)
print(acc)
