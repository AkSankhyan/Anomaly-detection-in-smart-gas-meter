# src/evaluate_model.py 

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

import numpy as np
from tensorflow.keras.models import load_model
from data_preprocessing import load_data, preprocess_data, split_data
from sklearn.metrics import mean_squared_error
from keras.losses import MeanSquaredError  # Import MeanSquaredError

# Load the models
cnn_autoencoder = load_model('models/cnn_autoencoder.h5', custom_objects={'mse': MeanSquaredError()})
# Assuming lstm_autoencoder is defined similarly
lstm_autoencoder = load_model('models/lstm_autoencoder.h5')

# Load and preprocess data
data = load_data('data/Processed Data/1002.csv')
features, target = preprocess_data(data)
X_train, X_test, y_train, y_test = split_data(features, target)

# Reshape data for CNN and LSTM
X_test_cnn = np.expand_dims(X_test, axis=2)  # CNN expects input shape [batch_size, timesteps, features]
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))  # LSTM expects [batch_size, timesteps, features]

# Evaluate the models
cnn_reconstructions = cnn_autoencoder.predict(X_test_cnn)
lstm_reconstructions = lstm_autoencoder.predict(X_test_lstm)

# Calculate the Mean Squared Error for anomaly detection
cnn_mse = np.mean(np.power(X_test_cnn - cnn_reconstructions, 2), axis=1)
lstm_mse = np.mean(np.power(X_test_lstm - lstm_reconstructions, 2), axis=1)

# Set anomaly threshold (can be fine-tuned)
cnn_threshold = np.percentile(cnn_mse, 95)
lstm_threshold = np.percentile(lstm_mse, 95)

# Detect anomalies
cnn_anomalies = cnn_mse > cnn_threshold
lstm_anomalies = lstm_mse > lstm_threshold

# Save results to output folder
np.savetxt('output/cnn_anomalies.csv', cnn_anomalies, delimiter=',')
np.savetxt('output/lstm_anomalies.csv', lstm_anomalies, delimiter=',')

