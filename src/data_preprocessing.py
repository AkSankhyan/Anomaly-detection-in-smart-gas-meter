# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the data:
    1. Handle missing values
    2. Normalize the data
    """
    # Fill missing values with 0 (or other logic)
    df.fillna(0, inplace=True)

    # Convert timestamp to datetime if necessary
    # df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create features and target
    features = df.drop(columns=['RowIndex'],errors='ignore')  # Adjust 'target_column'
    target = df['RowIndex']  # Replace with actual column name


    # Normalize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, target

def split_data(features, target):
    """
    Split the data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
