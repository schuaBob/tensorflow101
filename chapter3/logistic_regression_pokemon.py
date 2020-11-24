import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

# load dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, 'dataset', 'pokemon.csv')
pokemon_df = pd.read_csv(data_path)
pokemon_df = pokemon_df.set_index('#')
data_path = os.path.join(BASE_DIR, 'dataset', 'combats.csv')
combat_df = pd.read_csv(data_path)

# fill NaN and change type
pokemon_df['Type 2'].fillna('empty', inplace=True)
pokemon_df['Type 1'] = pokemon_df['Type 1'].astype('category')
pokemon_df['Type 2'] = pokemon_df['Type 2'].astype('category')
pokemon_df['Legendary'] = pokemon_df['Legendary'].astype('int')

# to one-hot encoding
df_type1_one_hot = pd.get_dummies(pokemon_df['Type 1'])
df_type2_one_hot = pd.get_dummies(pokemon_df['Type 2'])
combine_df_one_hot = df_type1_one_hot.add(df_type2_one_hot, fill_value=0).astype('int64')
pd.options.display.max_columns = 30
pokemon_df = pokemon_df.join(combine_df_one_hot)

# type to 0,1,2,...18
pokemon_df['Type 1'] = pokemon_df['Type 1'].cat.codes
pokemon_df['Type 2'] = pokemon_df['Type 2'].cat.codes
pokemon_df.drop('Name', axis='columns', inplace=True)
combat_df['Winner'] = combat_df.apply(lambda x: 0 if x.Winner == x.First_pokemon else 1, axis='columns')

# split data
data_num = combat_df.shape[0]
indexes = np.random.permutation(data_num)
train_indexes = indexes[:int(data_num*0.6)]
val_indexes = indexes[int(data_num*0.6):int(data_num*0.8)]
test_indexes = indexes[int(data_num*0.8):]
train_data = combat_df.loc[train_indexes]
val_data = combat_df.loc[val_indexes]
test_data = combat_df.loc[test_indexes]

# Normalization
pokemon_df['Type 1'] = pokemon_df['Type 1'] / 19
pokemon_df['Type 2'] = pokemon_df['Type 2'] / 19
# every row and from column HP to column Generation
mean = pokemon_df.loc[:, 'HP':'Generation'].mean()
std = pokemon_df.loc[:, 'HP':'Generation'].std()
pokemon_df.loc[:, 'HP':'Generation'] = (pokemon_df.loc[:, 'HP':'Generation'] - mean) / std

# transform to numpy type
x_train_index = np.array(train_data.drop('Winner', axis='columns'))
x_val_index = np.array(val_data.drop('Winner', axis='columns'))
x_test_index = np.array(test_data.drop('Winner', axis='columns'))
y_train = np.array(train_data['Winner'])
y_val = np.array(val_data['Winner'])
y_test = np.array(test_data['Winner'])

# train type 1: numeric
pokemon_data_normal = np.array(pokemon_df.loc[:,:'Legendary'])
x_train_normal = pokemon_data_normal[x_train_index-1].reshape((-1, 20))
x_val_normal = pokemon_data_normal[x_val_index-1].reshape((-1, 20))
x_test_normal = pokemon_data_normal[x_test_index-1].reshape((-1, 20))

# train type 2: one-hot
pokemon_data_one_hot = np.array(pokemon_df.loc[:,'HP':])
x_train_one_hot = pokemon_data_one_hot[x_train_index-1].reshape((-1, 54))
x_val_one_hot = pokemon_data_one_hot[x_val_index-1].reshape((-1, 54))
x_test_one_hot = pokemon_data_one_hot[x_test_index-1].reshape((-1, 54))

# build model 1
inputs = keras.Input(shape=(20,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(16, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model_name = 'pokemon_model_1'
model_1 = keras.Model(inputs = inputs, outputs = outputs, name=model_name)
model_1.summary()
model_1.compile(
    optimizer=keras.optimizers.Adam(), 
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()]
)

model_dir = os.path.join(BASE_DIR, 'models')
log_dir = os.path.join(BASE_DIR, 'logs', model_name)
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
# the larger the accuracy, the better
model_mckp = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(model_dir, f'{model_name}.h5'),
    monitor='val_binary_accuracy',
    save_best_only=True,
    mode='max'
)
history_1 = model_1.fit(x_train_normal, y_train, batch_size=64, epochs=200, validation_data=(x_val_normal, y_val), callbacks=[model_cbk, model_mckp])


# build model 2
inputs = keras.Input(shape=(54,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(16, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model_name = 'pokemon_model_2'
model_2 = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
model_2.summary()
model_2.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()]
)
log_dir = os.path.join(BASE_DIR, 'logs', model_name)
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
# the larger the accuracy, the better
model_mckp = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(model_dir, f'{model_name}.h5'),
    monitor='val_binary_accuracy',
    save_best_only=True,
    mode='max'
)
history_2 = model_2.fit(x_train_one_hot, y_train, batch_size=64, epochs=200, validation_data=(
    x_val_one_hot, y_val), callbacks=[model_cbk, model_mckp])

model_1.load_weights(os.path.join(model_dir, 'pokemon_model_1.h5'))
model_2.load_weights(os.path.join(model_dir, 'pokemon_model_2.h5'))
loss_1, accuracy_1 = model_1.evaluate(x_test_normal, y_test)
loss_2, accuracy_2 = model_2.evaluate(x_test_one_hot, y_test)
print('Model with numeric: {}%\n Model with one-hot: {}%'.format(accuracy_1, accuracy_2))

venusaur = np.expand_dims(pokemon_data_one_hot[3], axis=0)
charizard = np.expand_dims(pokemon_data_one_hot[7], axis=0)
blastoise = np.expand_dims(pokemon_data_one_hot[12], axis=0)

pred = model_2.predict(np.concatenate([venusaur, charizard], axis=-1))
winner = '妙蛙花' if pred < 0.5 else '噴火龍'
print(f'pred={pred}, {winner} 獲勝')

pred = model_2.predict(np.concatenate([charizard, blastoise], axis=-1))
winner = '噴火龍' if pred < 0.5 else '水箭龜'
print(f'pred={pred}, {winner} 獲勝')

pred = model_2.predict(np.concatenate([blastoise, venusaur], axis=-1))
winner = '水箭龜' if pred < 0.5 else '妙蛙花'
print(f'pred={pred}, {winner} 獲勝')
