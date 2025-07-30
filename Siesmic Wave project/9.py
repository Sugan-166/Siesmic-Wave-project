import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

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

# Evaluate the model on new data
def evaluate_model(model, data):
    predictions = model.predict(data)
    return predictions

# Save predictions to CSV
def save_predictions(predictions, output_file):
    df = pd.DataFrame(predictions, columns=['Prediction'])
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Main function
if __name__ == "__main__":
    model_path = r'D:\Earthquak_project\model.h5'  # Path to your saved model
    features_file = r'D:\Earthquak_project\features.csv'  # Path to your new data
    output_file = r'D:\Earthquak_project\predictions.csv'  # Path to save predictions
    
    # Load the trained model
    model = load_trained_model(model_path)
    
    # Load and preprocess new data
    try:
        # Load the original training features to fit the scaler
        training_features_file = r'D:\Earthquak_project\features.csv'  # Path to the training features CSV
        training_features = pd.read_csv(training_features_file)
        if training_features.empty:
            raise ValueError("The training features file is empty.")
        
        # Fit the scaler on training features
        scaler = StandardScaler()
        scaler.fit(training_features)
        
        # Preprocess new data
        new_data = preprocess_new_data(features_file, scaler)
        
        # Evaluate the model
        predictions = evaluate_model(model, new_data)
        
        # Save predictions
        save_predictions(predictions, output_file)

    except Exception as e:
        print(f"An error occurred: {e}")
