import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('xception_v1_02_1.000.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('fruit-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)

