import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
x = tf.data.Dataset.range(10)
y = tf.data.Dataset.range(10).map(lambda x: x*2)
dataset = tf.data.Dataset.zip({'x':x, 'y':y}).batch(2)
dataset = dataset.shuffle(10)
dataset = dataset.repeat(3)
for d in dataset:
    print('x:{}, y:{}'.format(d['x'], d['y']))