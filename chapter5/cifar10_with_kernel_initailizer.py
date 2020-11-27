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
from .preprocessing import parse_fn, parse_aug_fn
BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def build_and_train_model(run_name, init):
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=init)(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu',
                      kernel_initializer=init)(x)
    x = layers.Conv2D(256, (3, 3), activation='relu',
                      kernel_initializer=init)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu',
                      kernel_initializer=init)(x)
    x = layers.Conv2D(64, (3, 3), activation='relu',
                      kernel_initializer=init)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu', kernel_initializer=init)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    log_files = os.path.join(BASEDIR, 'logs', run_name, init.__class__.__name__)
    # histogram_freq = 1, 每次epoch各層的網路權重分布都會被記錄下來
    model_cbk = keras.callbacks.TensorBoard(log_dir=log_files, histogram_freq=1)
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


session_num = 1
weights_initialization_list = [
    initializers.RandomNormal(0, 0.01),
    initializers.glorot_normal(),
    initializers.he_normal()
]
for init in weights_initialization_list:
    print('---Running training session %d' % (session_num))
    run_name = "run-%d" % session_num
    build_and_train_model(run_name, init)
    session_num+=1
losses, acces = [], []
for i in range(1,4):
    run_name = "run-%d" % i
    model = keras.models.load_model(os.path.join(BASEDIR, 'models', f'{run_name}.h5'))
    loss, acc = model.evaluate(test_data)
    losses.append(loss)
    acces.append(acc)

print(losses)
print(acces)
