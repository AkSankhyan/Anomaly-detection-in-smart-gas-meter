# src/lstm_autoencoder.py

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input

from keras.models import load_model
from keras.losses import MeanSquaredError

from tensorflow.keras import losses, saving

@tf.keras.saving.register_keras_serializable()
def mse(y_true, y_pred):
    return losses.mean_squared_error(y_true, y_pred)


# Define the loss function explicitly
cnn_autoencoder = load_model('models/cnn_autoencoder.h5', custom_objects={'mse': MeanSquaredError()})

def create_lstm_autoencoder(input_shape):
    """
    Create an LSTM-based Autoencoder model.
    """
    input_layer = Input(shape=input_shape)

    # Encoder
    x = LSTM(128, activation='relu', return_sequences=False)(input_layer)
    x = RepeatVector(input_shape[0])(x)

    # Decoder
    x = LSTM(128, activation='relu', return_sequences=True)(x)
    x = TimeDistributed(Dense(input_shape[1]))(x)

    # Create the autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=x)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder
