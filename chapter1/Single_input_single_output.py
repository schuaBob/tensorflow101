from tensorflow import keras
inputs = keras.Input(shape=(784,), name='Input')
h1 = keras.layers.Dense(64, activation='relu', name='hidden1')(inputs)
h2 = keras.layers.Dense(64, activation='relu', name='hidden2')(h1)
outputs = keras.layers.Dense(10, activation='softmax', name='Output')(h2)
model = keras.Model(inputs=inputs, outputs=outputs)
keras.utils.plot_model(model, to_file='Single_input_single_output.png', show_shapes=True, show_layer_names=True)

