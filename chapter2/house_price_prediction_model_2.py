import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
# load dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, 'dataset', 'kc_house_data.csv')
data = pd.read_csv(data_path)
pd.options.display.max_columns = 25

# change datatype
data['year'] = pd.to_numeric(data['date'].str.slice(0,4))
data['month'] = pd.to_numeric(data['date'].str.slice(4, 6))
data['day'] = pd.to_numeric(data['date'].str.slice(6, 8))
data.drop(['id'], axis='columns', inplace=True)
data.drop(['date'], axis='columns', inplace=True)

# split dataset
data_num = data.shape[0]
indexes = np.random.permutation(data_num)
train_indexes = indexes[:int(data_num*0.6)]
val_indexes = indexes[int(data_num*0.6):int(data_num*0.8)]
test_indexes = indexes[int(data_num*0.8):]
train_data = data.loc[train_indexes]
val_data = data.loc[val_indexes]
test_data = data.loc[test_indexes]

# Normalization
train_validation_data = pd.concat([train_data, val_data])
mean = train_validation_data.mean()
std = train_validation_data.std()
# to z-score
train_data = (train_data - mean) / std
val_data = (val_data - mean) /std

# transform data to numpy
x_train = np.array(train_data.drop('price', axis='columns'))
y_train = np.array(train_data['price'])
x_val = np.array(val_data.drop('price', axis='columns'))
y_val = np.array(val_data['price'])

# build model
model = keras.Sequential(name='model-2')
model.add(layers.Dense(16, activation='relu', input_shape=(21,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))
model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsoluteError()]
) # loss is for model use, and metrics is for people to see how good the model is
model_name = 'Best_model_2'
model_dir = os.path.join(BASE_DIR, 'models')
log_dir = os.path.join(BASE_DIR, 'logs', model_name)
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
model_mckp = keras.callbacks.ModelCheckpoint(
    os.path.join(model_dir, f'{model_name}.h5'),
    monitor='val_mean_absolute_error',
    save_best_only=True,
    mode='min'
)

history = model.fit(x_train, y_train, batch_size=64, epochs=300, validation_data=(x_val, y_val), callbacks=[model_cbk, model_mckp])
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='prediction')
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend(loc='upper right')
# plt.show()
model.load_weights(os.path.join(BASE_DIR, 'models', f'{model_name}.h5'))
y_test = np.array(test_data['price'])
test_data = (test_data - mean) / std
x_test = np.array(test_data.drop('price', axis='columns'))
y_predict = model.predict(x_test)
print(y_predict)
y_predict = np.reshape(y_predict * std['price'] + mean['price'], y_test.shape)
print(y_predict)
percentage_error = np.mean(np.abs(y_test - y_predict)) / np.mean(y_test) * 100
print('Model_1 Percentage Error: {:.2f}%'.format(percentage_error))
