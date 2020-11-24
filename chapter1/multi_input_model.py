from tensorflow import keras
img_input = keras.layers.Input(shape=(128,128,3), name='Image_Input')
info_input = keras.layers.Input(shape=(1,), name='Information_Input')

h1_1 = keras.layers.Conv2D(64, 5, strides=2, activation='relu', name='hidden1_1')(img_input)
h1_2 = keras.layers.Conv2D(32, 5, strides=2, activation='relu', name='hidden1_2')(h1_1)
h1_2_ft = keras.layers.Flatten()(h1_2)

h1_3 = keras.layers.Dense(64, activation='relu', name='hidden1_3')(info_input)
concat = keras.layers.Concatenate()([h1_2_ft, h1_3])
h2 = keras.layers.Dense(64, activation='relu', name='hidden2')(concat)
# output price forecast
outputs = keras.layers.Dense(1, name='Output')(h2)

model = keras.Model(inputs=[img_input, info_input], outputs=outputs)
keras.utils.plot_model(model, to_file='function_api_multi_input_model.png', show_layer_names=True, show_shapes=True)
