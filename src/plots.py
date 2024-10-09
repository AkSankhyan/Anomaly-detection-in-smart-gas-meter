import pandas as pd
import matplotlib.pyplot as plt

# Load anomalies
cnn_anomalies = pd.read_csv('output/cnn_anomalies.csv', header=None)
lstm_anomalies = pd.read_csv('output/lstm_anomalies.csv', header=None)

# Plotting the anomalies detected by CNN
plt.figure(figsize=(12, 6))
plt.plot(cnn_anomalies, label='CNN Anomalies', color='red')
plt.title('CNN Anomalies Detected')
plt.xlabel('Sample Index')
plt.ylabel('Anomaly Detected (1 = Yes, 0 = No)')
plt.legend()
plt.show()

# Plotting the anomalies detected by LSTM
plt.figure(figsize=(12, 6))
plt.plot(lstm_anomalies, label='LSTM Anomalies', color='blue')
plt.title('LSTM Anomalies Detected')
plt.xlabel('Sample Index')
plt.ylabel('Anomaly Detected (1 = Yes, 0 = No)')
plt.legend()
plt.show()
