# src/cnn_autoencoder.py

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras import losses


from keras.models import load_model
from keras.losses import MeanSquaredError

from tensorflow.keras import losses, saving

@tf.keras.saving.register_keras_serializable()
def mse(y_true, y_pred):
    return losses.mean_squared_error(y_true, y_pred)


# Define the loss function explicitly
cnn_autoencoder = load_model('models/cnn_autoencoder.h5', custom_objects={'mse': MeanSquaredError()})


def create_cnn_autoencoder(input_shape):
    """
    Create a CNN-based Autoencoder model.
    """
    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(16, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Decoder
    x = Conv1D(16, kernel_size=3, activation='relu', padding='same')(x)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = UpSampling1D(size=2)(x)

    output_layer = Conv1D(1, kernel_size=3, activation='sigmoid', padding='same')(x)

    # Create the autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder
