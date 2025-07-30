# Testing

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the trained model
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# Preprocess new data
def preprocess_new_data(features_file, scaler):
    try:
        features = pd.read_csv(features_file)
        if features.empty:
            raise ValueError("The features file is empty.")
        scaled_features = scaler.transform(features)
        return scaled_features
    except pd.errors.EmptyDataError:
        print("Error: The file is empty or not readable.")
        raise
    except ValueError as e:
        print(f"Error: {e}")
        raise

# Detect and alert based on predictions
def detect_and_alert(model, data):
    predictions = model.predict(data)
    # Assuming binary classification: if prediction > 0.5, alert
    alerts = predictions > 0.5
    return alerts

# Main function
if __name__ == "__main__":
    model_path = r'D:\Earthquak_project\model.h5'  # Path to your saved model
    features_file = r'D:\Earthquak_project\features.csv'  # Path to your new data

    # Load the trained model
    model = load_trained_model(model_path)
    
    # Load and preprocess new data
    try:
        scaler = StandardScaler()  # Use the same scaler that was used during training
        # Load the original training features to fit the scaler
        training_features_file = r'D:\Earthquak_project\features.csv'  # Path to the training features CSV
        features = pd.read_csv(training_features_file)
        if features.empty:
            raise ValueError("The training features file is empty.")
        scaler.fit(features)  # Fit scaler on training features

        new_data = preprocess_new_data(features_file, scaler)
    
        # Detect and generate alerts
        alerts = detect_and_alert(model, new_data)
    
        # Output alerts
        for i, alert in enumerate(alerts):
            if alert:
                print(f"Alert: Anomaly detected in sample {i + 1}")
            else:
                print(f"Sample {i + 1}: No anomaly detected")

    except Exception as e:
        print(f"An error occurred: {e}")
