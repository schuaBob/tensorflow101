import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Input, Model

image_inputs = Input(shape=(256,256,3),name='Image_Input')
info_inputs = Input(shape=(10,), name='Info_Input')

h1 = Conv2D(64, 3, activation='relu', name='hidden1')(image_inputs)
h2 = Conv2D(64, 3, activation='relu', name='hidden2')(h1)
h3 = Conv2D(64, 3, activation='relu', name='hidden3')(h2)
flatten = Flatten()(h3)

h4 = Dense(64)(info_inputs)
concat = Concatenate()([flatten, h4])
weather_outputs = Dense(1, name='Output1')(concat)
temp_outputs = Dense(1, name='Output2')(concat)
humidity_outputs = Dense(1, name='Output3')(concat)

model = Model(inputs=[image_inputs, info_inputs], outputs = [weather_outputs, temp_outputs, humidity_outputs])
plot_model(model, show_layer_names=True, show_shapes=True, to_file='multi_input_multi_output.png')
