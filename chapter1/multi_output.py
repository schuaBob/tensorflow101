from tensorflow import keras
inputs = keras.Input(shape=(128,128,3), name='Input')
h1 = keras.layers.Conv2D(64, 3, activation='relu', name='hidden1')(inputs)
h2 = keras.layers.Conv2D(64, 3, activation='relu', name='hidden2')(h1)
h3 = keras.layers.Conv2D(64, 3, activation='relu', name='hidden3')(h2)
flatten = keras.layers.Flatten()(h3)
age_output = keras.layers.Dense(1, name='Age_Output')(flatten)
gender_output = keras.layers.Dense(1, name='Gender_Output')(flatten)
model = keras.Model(inputs=inputs, outputs=[age_output, gender_output])
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='Multiple_output.png')
