# src/train_model.py

import os

# Set the environment variable to disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs

import tensorflow as tf

# Your existing code for loading data, building models, etc.
# For example:
# data = load_data("data/Processed_Data/1000.csv")
# model = build_cnn_model()  # or any other model function you have


import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from cnn_autoencoder import create_cnn_autoencoder
from lstm_autoencoder import create_lstm_autoencoder
from data_preprocessing import load_data, preprocess_data, split_data

# Load and preprocess data
data = load_data('data/Processed Data/1002.csv')
features, target = preprocess_data(data)
X_train, X_test, y_train, y_test = split_data(features, target)

# Reshape for CNN (if necessary)
X_train_cnn = np.expand_dims(X_train, axis=2)
X_test_cnn = np.expand_dims(X_test, axis=2)

# Reshape for LSTM (3D input)
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train CNN Autoencoder
cnn_autoencoder = create_cnn_autoencoder((X_train_cnn.shape[1], 1))
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
cnn_autoencoder.fit(X_train_cnn, X_train_cnn, epochs=50, batch_size=32, validation_data=(X_test_cnn, X_test_cnn), callbacks=[early_stopping])

# Train LSTM Autoencoder
lstm_autoencoder = create_lstm_autoencoder((X_train_lstm.shape[1], 1))
lstm_autoencoder.fit(X_train_lstm, X_train_lstm, epochs=50, batch_size=32, validation_data=(X_test_lstm, X_test_lstm), callbacks=[early_stopping])

# Save the models
cnn_autoencoder.save('models/cnn_autoencoder.h5')
lstm_autoencoder.save('models/lstm_autoencoder.h5') 
