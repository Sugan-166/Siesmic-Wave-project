#Validation

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load and preprocess new data
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
    return predictions

# Main function
if __name__ == "__main__":
    model_path = r'D:\Earthquak_project\model.h5'  # Path to your saved model
    features_file = r'D:\Earthquak_project\features.csv'  # Path to your new data
    
    # Load the trained model
    model = load_model(model_path)

    try:
        # Load the original training features to fit the scaler
        training_features_file = r'D:\Earthquak_project\features.csv'  # Path to the training features CSV
        training_features = pd.read_csv(training_features_file)
        if training_features.empty:
            raise ValueError("The training features file is empty.")
        
        # Preprocess the features
        scaler = StandardScaler()
        scaler.fit(training_features)  # Fit scaler on training data
        
        # Preprocess new data for testing
        new_data = preprocess_new_data(features_file, scaler)
        
        # Detect and generate predictions
        predictions = detect_and_alert(model, new_data)
        
        # Output predictions
        for i, prediction in enumerate(predictions):
            print(f"Sample {i+1}: Prediction = {prediction[0]}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
